import os
import os.path as osp
import random

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from tqdm import tqdm

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


# class OriginalVisionEncoder(nn.Module):
#     def __init__(self, cfg, clip_model):
#         super(OriginalVisionEncoder, self).__init__()
#         self.visual = clip_model.visual


class VisionEncoder(nn.Module):
    def __init__(self, cfg, clip_model):  # , image_weight
        super().__init__()
        visual = clip_model.visual  # CLIP's visual encoder
        self.ln_pre = visual.ln_pre
        self.transformer = visual.transformer.resblocks
        self.ln_post = visual.ln_post
        self.proj = visual.proj
        self.layers = len(self.transformer)
        # self.n_pro = cfg.TRAINER.META.N_PRO
        # self.layer_p = cfg.TRAINER.META.LAYERS
        self.n_pro = 12
        self.layer_p = 12
        # self.bias = nn.Parameter(torch.empty(self.layer_p, 1, 1, 768).uniform_(-1, 1).half())

    def forward(self, x, ctx_v):
        x = torch.cat([x, ctx_v[:, 0, :, :]], dim=1)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # 199 * 128 * 768
        # print(f"x size is {x.size()}")

        for i in range(self.layers):
            if 1 <= i < self.layer_p:
                ctx = ctx_v[:, i].permute(1, 0, 2)
                prefix = x[:-self.n_pro, :, :]
                x = torch.cat([prefix, ctx], dim=0)
                # x = x + self.bias
                # print(f"x size is {x.size()}")
                # x = x + self.bias[i]
            x = self.transformer[i](x)

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj

        return x


class VisionPromptLearner(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        n_pro = 12
        self.dtype = clip_model.dtype
        ctx_dim = clip_model.visual.ln_pre.weight.shape[0]
        self.visual = clip_model.visual
        self.conv1 = self.visual.conv1
        self.class_embedding = self.visual.class_embedding
        self.positional_embedding = self.visual.positional_embedding
        self.layers = len(self.visual.transformer.resblocks)
        # self.layer_p = cfg.TRAINER.META.LAYERS
        self.layer_p = 12

        # domain-invariant and domain-specific visual prompts
        ctx_vectors = torch.empty(self.layer_p, n_pro, ctx_dim, dtype=self.dtype)
        ctx_vectors_domain = torch.empty(self.layer_p, n_pro, ctx_dim, dtype=self.dtype)

        # kaiming 初始化
        nn.init.normal_(ctx_vectors, std=0.02)
        nn.init.normal_(ctx_vectors_domain, std=0.02)

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        self.ctx_domain = nn.Parameter(ctx_vectors_domain)  # to be optimized

    def forward(self, x):
        ctx = self.ctx
        if ctx.dim() == 3:
            ctx = ctx.unsqueeze(0).expand(len(x), -1, -1, -1)

        # x = self.conv1(x.type(self.dtype))  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)
        x = x + self.positional_embedding.type(self.dtype)

        return x, ctx


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        prefix = self.token_prefix
        suffix = self.token_suffix

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts


class PromptLearner_domain(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        print(f"lens of domain token{n_cls}")
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts for domain")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context for domain")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context for domain: "{prompt_prefix}"')
        print(f"Number of context words (tokens) for domain: {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        prefix = self.token_prefix
        suffix = self.token_suffix

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, domain, clip_model, device):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.prompt_learner_domain = PromptLearner_domain(cfg, domain, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.tokenized_prompts_domain = self.prompt_learner_domain.tokenized_prompts
        self.vision_prompt_learner = VisionPromptLearner(cfg, clip_model)
        self.image_encoder = VisionEncoder(cfg, clip_model)
        self.image_encoder_clip = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.move_step = cfg.TRAINER.STEP
        # self.num_mix = len(domain)
        self.num_mix = 3
        self.device = device

    def calc_mean_std(self, feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert len(size) == 4
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def shuffle_data_chose_domain(self, data, domain=None, chosen_idx=2):
        batch_size = data.size(0)
        data_accept = data[domain == chosen_idx]
        num_accept = data_accept.size(0)
        if num_accept == 0:
            return self.shuffle_data(data)
        random_index = torch.randint(0, num_accept, (batch_size,), device=self.device)
        new_data = torch.index_select(data_accept, dim=0, index=random_index)
        return new_data

    def shuffle_data(self, data):
        num = data.size(0)
        index = torch.randint(0, num, (num,), device=self.device)
        new_data = torch.index_select(data, dim=0, index=index)
        return new_data

    def prepare(self, batch_image, batch_domain=None, num_mix=None):
        image_features = self.image_encoder_clip.conv1(batch_image.type(self.dtype))
        features_size = image_features.size()
        batch_mean, batch_std = self.calc_mean_std(image_features)
        # normalized the feature map
        normalized_image_feature = (image_features - batch_mean.expand(features_size)) / batch_std.expand(features_size)
        # concate mean and std
        mean_std = torch.cat([batch_mean.squeeze().unsqueeze(1), batch_std.squeeze().unsqueeze(1)], dim=1)
        # use num_mix to get mean_std_set
        mean_std_set = torch.cat(
            [mean_std.unsqueeze(1)] + [self.shuffle_data_chose_domain(mean_std, batch_domain, idx).unsqueeze(1) for
                                       idx in range(num_mix - 1)], dim=1)
        return mean_std_set.detach(), image_features.detach(), normalized_image_feature.detach()

    def init_weight(self, batch_size, num_mix):
        # init weight
        ori_weight = torch.ones((batch_size, num_mix), device=self.device) / num_mix
        return ori_weight

    def stylize(self, mix_weight, mean_std_set, x_featuremap, normalized_x_featuremap, gamma=1.0):
        featuremap_size = x_featuremap.size()
        mean_std_set_size = mean_std_set.size()

        # # normalized the weight
        weight_norm = mix_weight / (torch.sum(mix_weight, dim=1, keepdim=True) + 1e-6)  # N,num_mix
        weight_norm = weight_norm.type(mean_std_set.type())

        # calc mean_std mix
        mean_std_mix = torch.bmm(weight_norm.unsqueeze(1),
                                 mean_std_set.view(mean_std_set_size[0], mean_std_set_size[1], -1)).view(
            mean_std_set_size[0], mean_std_set_size[2], mean_std_set_size[3])

        # chunk mean_std to mean and std
        mean_mix, std_mix = torch.chunk(mean_std_mix, 2, dim=1)  # N,1,C

        mean_mix = mean_mix.squeeze(1).unsqueeze(2).unsqueeze(2)  # N,C,1,1
        std_mix = std_mix.squeeze(1).unsqueeze(2).unsqueeze(2)  # N,C,1,1

        # apply mean and std
        x_featuremap_mix = normalized_x_featuremap * std_mix.expand(featuremap_size) + mean_mix.expand(featuremap_size)

        # use gamma calc gamma mix featuremap
        x_featuremap_final = x_featuremap_mix * gamma + x_featuremap * (1 - gamma)

        return x_featuremap_final, weight_norm

    def forward(self, image, label=None, domain=None, optim=None, mix_weight=None):
        if self.training:
            # text feature generate
            prompts = self.prompt_learner()
            prompts_domain = self.prompt_learner_domain()
            tokenized_prompts = self.tokenized_prompts
            tokenized_prompts_domain = self.tokenized_prompts_domain
            text_features = self.text_encoder(prompts, tokenized_prompts)
            text_features_domain = self.text_encoder(prompts_domain, tokenized_prompts_domain)

            # normalization
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features_domain = text_features_domain / text_features_domain.norm(dim=-1, keepdim=True)

            logit_scale = self.logit_scale.exp()
            # get mean and std
            mean_std_set, x_featuremap, normalized_x_featuremap = self.prepare(image, domain, self.num_mix)
            if mix_weight is None:
                # initialize the alpha
                ori_weight = self.init_weight(x_featuremap.size(0), self.num_mix)
                mix_weight = ori_weight.detach_()
                for i in range(self.move_step):
                    # update mix weight
                    mix_weight.requires_grad = True
                    self.vision_prompt_learner.ctx.requires_grad = False
                    x_fake, weight_norm = self.stylize(mix_weight, mean_std_set, x_featuremap, normalized_x_featuremap)
                    x_fake, ctx_v_fake = self.vision_prompt_learner(x_fake)
                    image_features_fake = self.image_encoder(x_fake, ctx_v_fake)
                    image_features_fake = image_features_fake / image_features_fake.norm(dim=-1, keepdim=True)

                    x, ctx_v = self.vision_prompt_learner(x_featuremap)
                    image_features = self.image_encoder(x, ctx_v)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                    score_fake = logit_scale * image_features_fake @ text_features.t()
                    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-07)
                    score = cos(image_features, image_features_fake)
                    score = 1.0 - torch.mean(score)

                    lambda_loss = 8.0

                    loss = F.cross_entropy(score_fake, label)
                    loss_all = loss - lambda_loss * score
                    loss_all.backward()

                    # update weight
                    mu = 0.05
                    mix_weight = mix_weight + mu * mix_weight.grad.sign()
                    eta = torch.clamp(mix_weight - ori_weight, min=-1, max=1)
                    mix_weight = torch.clamp(ori_weight + eta, min=0, max=1).detach_()

                optim.zero_grad()
                mix_weight.requires_grad = False
                self.vision_prompt_learner.ctx.requires_grad = True

            # freeze the mix weight
            x_fake, weight_norm = self.stylize(mix_weight, mean_std_set, x_featuremap, normalized_x_featuremap)
            x_fake, ctx_v_fake = self.vision_prompt_learner(x_fake)
            image_features_fake = self.image_encoder(x_fake, ctx_v_fake)
            image_features_fake = image_features_fake / image_features_fake.norm(dim=-1, keepdim=True)
            score_fake = logit_scale * image_features_fake @ text_features.t()

            # image feature generate
            x, ctx_v = self.vision_prompt_learner(x_featuremap)
            image_features = self.image_encoder(x, ctx_v)
            image_features_ori = self.image_encoder_clip(image.type(self.dtype))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            image_features_ori = image_features_ori / image_features_ori.norm(dim=-1, keepdim=True)

            cos = torch.nn.CosineSimilarity(dim=1, eps=1e-07)
            score = cos(image_features, image_features_ori)
            score = 1.0 - torch.mean(score)

            logits1 = logit_scale * image_features @ text_features.t()

            return logits1, score, score_fake, mix_weight
        else:
            prompts = self.prompt_learner()
            prompts_domain = self.prompt_learner_domain()
            tokenized_prompts = self.tokenized_prompts
            tokenized_prompts_domain = self.tokenized_prompts_domain
            text_features = self.text_encoder(prompts, tokenized_prompts)
            text_features_domain = self.text_encoder(prompts_domain, tokenized_prompts_domain)

            # normalization
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features_domain = text_features_domain / text_features_domain.norm(dim=-1, keepdim=True)

            # image feature generate
            x_featuremap = self.image_encoder_clip.conv1(image.type(self.dtype))
            x, ctx_v = self.vision_prompt_learner(x_featuremap)
            image_features = self.image_encoder(x, ctx_v)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            logit_scale = self.logit_scale.exp()
            logits1 = logit_scale * image_features @ text_features.t()

            return logits1


class Norm:
    """Normalize batch images."""

    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor(std).cuda().view(1, 3, 1, 1)

    def __call__(self, tensor):
        """
        Input:
            tensor (torch.Tensor): tensor image of size (B, C, H, W)
        """
        tensor -= self.mean
        tensor /= self.std
        return tensor


@TRAINER_REGISTRY.register()
class CoOp(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        domain = cfg.DATASET.SOURCE_DOMAINS
        self.num_mix = len(domain)
        print("all of the domain is: ")
        print(domain)

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, domain, clip_model, device)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "vision_prompt_learner.ctx" not in name:
                param.requires_grad_(False)

        # pre-trained load
        all_domain = ['labelme', 'pascal', 'sun', 'caltech']
        for each_domain in all_domain:
            if each_domain == cfg.DATASET.TARGET_DOMAINS[0]:
                weight_dir = f"tpami/vlcs/CoOp_train_image_8.0_npro_12_batch_16_warmup_10_{each_domain[0]}/vit_b16_16shots/nctx16_cscFalse_ctpend/seed2024/model/model-best.pth.tar"
                load_pretrained_weights(self.model, weight_dir)

        self.device = device
        self.model.to(self.device)
        self.num_batches = len(self.train_loader_x)
        print(f"num batches is {self.num_batches}")
        self.mix_weight = torch.zeros(self.num_batches, 16, 3).to(self.device)

        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("model", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

    def forward_backward(self, batch):
        image, label, domain = self.parse_batch_train(batch)
        if self.epoch == 0:
            output1, output2, output3, weight = self.model(image, label, domain, self.optim)
            self.mix_weight[self.batch_idx] = weight
        else:
            output1, output2, output3, weight = self.model(image, label, domain, self.optim,
                                                           self.mix_weight[self.batch_idx])
        alf_loss = 8.0
        beta_loss = 0.4
        loss1 = F.cross_entropy(output1, label)
        loss3 = F.cross_entropy(output3, label)
        loss = (1 - beta_loss) * loss1 + alf_loss * output2 + beta_loss * loss3
        self.model_backward_and_update(loss)

        loss_summary = {
            "loss1": loss1.item(),
            "loss2": output2.item(),
            "loss3": loss3.item(),
            "acc1": compute_accuracy(output1, label)[0].item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        domain = batch["domain"]

        input = input.to(self.device)
        label = label.to(self.device)
        domain = domain.to(self.device)

        return input, label, domain

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
