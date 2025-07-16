import os
import random
import argparse
import yaml
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms

from datasets import build_dataset
from datasets.utils import build_data_loader
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from utils import *
from torch.autograd import Variable
from torch.distributions import uniform

from bbdm_clip import GaussianDiffusion1D
from bbdm_model import Unet1D
from ema import EMA

_tokenizer = _Tokenizer()
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
])

hard_attribute = ['Primary petal color','Secondary petal color','Petal shape','Petal texture',
                  'Petal edge','Number of petals','Flower center color','Flower center texture',
                  'Petal arrangement','Overall color contrast']

class CoOp_PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        # n_cls = len(classnames)
        n_ctx = 4
        ctx_init = 'a photo of a' # caltech101

        self.dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        # self.n_cls = n_cls
        self.n_ctx = n_ctx

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init).cuda()
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(self.dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :].cuda()
            prompt_prefix = ctx_init
            self.n_ctx = n_ctx
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=self.dtype).cuda()
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)
        self.prompt_prefix = prompt_prefix
        # self.get_prefix_suffix_token(classnames, clip_model)

    def get_prefix_suffix_token(self, classnames, clip_model):
        n_cls = len(classnames)
        self.n_cls = n_cls
        prompt_prefix = self.prompt_prefix
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(self.dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx :, :])  # CLS, EOS

        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512)
        )
        self.apply(weights_init)

    def forward(self, x):
        essential = self.net(x)
        return essential

class query_attribute_value(nn.Module):
    def __init__(self, hard_prompt):
        super(query_attribute_value, self).__init__()
        self.prompt = hard_prompt
        self.prompt = nn.Parameter(self.prompt, requires_grad=True)
        self.to_q = nn.Linear(512, 512)
        self.to_k = nn.Linear(512, 512)
        self.to_v = nn.Linear(512, 512)
        self.out = nn.Linear(512, 512)
        self.scale = 2 ** -0.5
        self.apply(weights_init)

    def forward(self, attribute):
        b = attribute.size()[0]
        q = self.to_q(self.prompt)
        k, v = self.to_k(attribute), self.to_v(attribute)
        # out = torch.zeros(b, self.num_p, 512).to(device)
        # for id in range(b):
        #     k_i, v_i = k[id].unsqueeze(0), v[id].unsqueeze(0)
        #     sim_i = torch.einsum('a d, y d -> a y', q, k_i) * self.scale
        #     attn_i = sim_i.softmax(dim=-1)
        #     out_i = torch.einsum('a y, y d -> a d', attn_i, v_i)
        #     out[id] = self.out(out_i)
        k = k.unsqueeze(1).repeat(1, b, 1)
        sim = torch.einsum('a d, n b d -> n a b', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = torch.einsum('n a b, b d -> n a d', attn, v)
        out = self.out(out)
        return out


class Decoder(nn.Module):
    def __init__(self, hard_prompt):
        super(Decoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        self.query = query_attribute_value(hard_prompt)
        self.apply(weights_init)

    def forward(self, essential, attribute):
        b = essential.size()[0]
        device = essential.device
        global_att = attribute
        visual_attribute_value = self.query(attribute)
        a = visual_attribute_value.size()[1]
        sample_matrix = torch.zeros(b, b + a, 1024).to(device)
        for i in range(b):
            image_sample = essential[i]
            for j in range(b):
                global_attribute_sample = global_att[j]
                global_sample = torch.cat((image_sample, global_attribute_sample), dim=0)
                sample_matrix[i, j, :] = global_sample
            for k in range(a):
                local_attribute_sample = visual_attribute_value[i, k, :]
                local_sample = torch.cat((image_sample, local_attribute_sample), dim=0)
                sample_matrix[i, b + k, :] = local_sample

        input = sample_matrix.resize(b * (b + a), 1024)
        output = self.net(input)
        output = output.resize(b, (b + a), 512)
        return output

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
        x = self.ln_final(x).float()

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

def entropy(prediction):
    b, c = prediction.size()[0], prediction.size()[1]
    device = prediction.device
    target_distribution = torch.full((c,), 1.0 / c)
    target_probs = target_distribution.to(device).unsqueeze(0).expand_as(prediction)
    kl_div = torch.sum(prediction * torch.log(prediction / target_probs), dim=1)
    loss = torch.mean(kl_div)
    return loss

def evaluate(generate, original, num_class):
    original = original.unsqueeze(1).repeat(1, num_class, 1)
    rec_logits = (generate - original).pow(2).sum(2)
    return 1 / rec_logits

def contrastive(generate, original, margin = 1.0):
    b = generate.size()[0]
    device = generate.device
    original = original.unsqueeze(1).repeat(1, b, 1).reshape(b * b, -1)
    generate = generate.reshape(b * b, -1)
    label = torch.eye(b).to(device)
    label = label.reshape(b * b)
    distances = torch.norm(generate - original, p=2, dim=1)
    positive_loss = label * (distances ** 2)
    negative_loss = (1 - label) * torch.clamp(margin - distances, min=0) ** 2
    loss = torch.mean(positive_loss + negative_loss)
    return loss

def run_causal(cfg, netE, netA, clip_model, preprocess, device, hard_prompt):
    text_encoder = TextEncoder(clip_model).float().to(device)
    prompt_learner = CoOp_PromptLearner(all_classnames, clip_model).to(device)
    optimizer_p = torch.optim.SGD(prompt_learner.parameters(), lr=2e-3)
    scheduler_p = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_p, cfg['train_epoch'] * len(train_loader_F))

    netD = Decoder(hard_prompt).to(device)
    optimizer_d = torch.optim.SGD(netD.parameters(), lr=2e-3)
    scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_d, cfg['train_epoch'] * len(train_loader_F))

    Query = query_attribute_value(hard_prompt).to(device)
    optimizer_q = torch.optim.SGD(Query.parameters(), lr=2e-3)
    scheduler_q = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_q, cfg['train_epoch'] * len(train_loader_F))

    softmax = nn.Softmax()
    CrossEntropy = nn.CrossEntropyLoss()

    best_base = 0.0
    best_new = 0.0

    for train_idx in range(0, 50 + 1):
        prompt_learner.train()
        netD.train()
        Query.train()
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, cfg['train_epoch']))

        for i, (images, target) in enumerate(tqdm(train_loader_F)):
            images, target = images.to(device), target.to(device)
            with torch.no_grad():
                image_features = clip_model.encode_image(images).float()
                image_features /= image_features.norm(dim=-1, keepdim=True)
                essential = netE(image_features)
                attribute = netA(image_features)
                b = image_features.size()[0]

            prompt_learner.zero_grad()
            netD.zero_grad()
            Query.zero_grad()

            prompt_learner.get_prefix_suffix_token(all_classnames, clip_model)
            prompts = prompt_learner()
            tokenized_prompts = prompt_learner.tokenized_prompts
            prompt_text_features = text_encoder(prompts, tokenized_prompts)
            prompt_text_features = prompt_text_features / prompt_text_features.norm(dim=-1, keepdim=True) # Cx512
            prompt_text_features = prompt_text_features.to(device)

            generate_sample = netD(essential, attribute) # bx10x512
            global_generate_sample = generate_sample[:, :b, :] # bxbx512
            local_generate_sample = generate_sample[:, b:, :] # bx10x512
            batch_text_features = prompt_text_features[target]
            batch_text_attribute_value = Query(batch_text_features) # bx10x512
            text_attribute_value = Query(prompt_text_features)
            a = local_generate_sample.size()[1]
            C = prompt_text_features.size()[0]
            loss_intra = 0.0
            for img_id in range(b):
                img_att = local_generate_sample[img_id] # 10x512
                text_att = batch_text_attribute_value[img_id] # 10x512
                logits = 100. * img_att.float() @ text_att.T.float() # 10x10
                target_att = torch.arange(0, a).to(device)
                loss_intra += CrossEntropy(logits, target_att)
                eyes = torch.eye(a).to(device)
                logits_inter = torch.zeros(b, C).to(device)
                for txt_id in range(C):
                    txt_att = text_attribute_value[txt_id]
                    logits_all = 100. * img_att.float() @ txt_att.T.float()
                    logits_all = logits_all * eyes
                    logits_inter[img_id, txt_id] = logits_all.sum()
            loss_inter = CrossEntropy(logits_inter, target)

            logits_original = 100. * image_features.float() @ prompt_text_features.T.float()
            loss_original = CrossEntropy(logits_original, target)

            index = torch.arange(0, b).to(device)
            real_generate = global_generate_sample[index, index, :] # bx512
            loss_rec = (real_generate - image_features).pow(2).sum(1).mean()

            loss_con = contrastive(global_generate_sample, image_features)

            label_fix_e_change_a = target.unsqueeze(1).repeat(1, b).reshape(b * b)
            logits_fix_e_change_a = 100. * global_generate_sample.reshape(b * b, -1).float() @ prompt_text_features.T.float()
            loss_fix_e_change_a = CrossEntropy(logits_fix_e_change_a, label_fix_e_change_a)
            label_change_e_fix_a = target.unsqueeze(0).repeat(b, 1).reshape(b * b)
            logits_change_e_fix_a = 100. * global_generate_sample.permute(1, 0, 2).reshape(b * b, -1).float() @ prompt_text_features.T.float()
            loss_change_e_fix_a = CrossEntropy(logits_change_e_fix_a, label_change_e_fix_a)

            loss_all = loss_original + loss_intra + loss_inter + loss_rec + loss_con + loss_fix_e_change_a + loss_change_e_fix_a
            loss_all.backward()
            loss_list.append(loss_all.item())
            optimizer_p.step()
            scheduler_p.step()
            optimizer_d.step()
            scheduler_d.step()
            optimizer_q.step()
            scheduler_q.step()

        print('Loss: {:.4f}\n'.format(sum(loss_list)/len(loss_list)))

        if train_idx % 5 == 0:
            prompt_learner.eval()
            netD.eval()
            Query.eval()

            # weights_path = "./checkpoint"
            # mkdir(weights_path)
            # torch.save(netE.state_dict(), os.path.join(weights_path, "netE_%d.pth" % (train_idx)))
            # torch.save(netA.state_dict(), os.path.join(weights_path, "netA_%d.pth" % (train_idx)))
            # torch.save(netD.state_dict(), os.path.join(weights_path, "netD_%d.pth" % (train_idx)))
            # torch.save(Query.state_dict(), os.path.join(weights_path, "query_%d.pth" % (train_idx)))
            # torch.save(prompt_learner.state_dict(), os.path.join(weights_path, "prompt_%d.pth" % (train_idx)))


            with torch.no_grad():
                prompt_learner.get_prefix_suffix_token(all_classnames, clip_model)
                prompts = prompt_learner()
                tokenized_prompts = prompt_learner.tokenized_prompts
                text_features = text_encoder(prompts, tokenized_prompts)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            #     text_features_base = text_features[:len(base_classnames)]
            #     text_attribute_base = Query(text_features_base)
            #     text_features_new = text_features[len(base_classnames):]
            #     text_attribute_new = Query(text_features_new)

            #     test_attribute_base = netA(test_features.float())
            #     test_attribute_base = netD.query(test_attribute_base)
            #     test_attribute_new = netA(test_features_new.float())
            #     test_attribute_new = netD.query(test_attribute_new)

            # eyes_base = torch.eye(a).to(device)
            # eyes_base = eyes_base.unsqueeze(0).repeat(test_features.size()[0], 1, 1).unsqueeze(1).repeat(1, text_features_base.size()[0], 1, 1)
            # logits_base = 100. * torch.einsum('b a d, c t d -> b c a t', test_attribute_base, text_attribute_base)
            # logits_base = logits_base * eyes_base
            # logits_base = logits_base.sum(dim=-1).sum(dim=-1)

            # eyes_new = torch.eye(a).to(device)
            # eyes_new = eyes_new.unsqueeze(0).repeat(test_features_new.size()[0], 1, 1).unsqueeze(1).repeat(1, text_features_new.size()[0], 1, 1)
            # logits_new = 100. * torch.einsum('b a d, c t d -> b c a t', test_attribute_new, text_attribute_new)
            # logits_new = logits_new * eyes_new
            # logits_new = logits_new.sum(dim=-1).sum(dim=-1)

            # new
            logits_new = 100. * test_features_new.float() @ text_features.T.float()[:, len(base_classnames):]
            new_acc = cls_acc(logits_new, test_labels_new)

            # base
            logits_base = 100. * test_features.float() @ text_features.T.float()[:, :len(base_classnames)]
            base_acc = cls_acc(logits_base, test_labels)

            H = 2 * base_acc * new_acc / (base_acc + new_acc)
            if new_acc > best_new:
                best_new = new_acc
            if base_acc > best_base:
                best_base = base_acc

            message = "epoch: %d base acc:\t%.2f  new acc:\t%.2f H:\t%.2f \n" % (train_idx, base_acc, new_acc, H)
            print(message)
            with open("results/+casual_pet.txt", "a") as f:
                f.write(message + '\n')

    message_best = "best base ass:\t%.2f best new acc:\t%.2f" % (best_base, best_new)
    print(message_best)
    with open("results/+casual_pet.txt", "a") as f:
        f.write(message_best + '\n')

def run_bbdm(cfg, dataset, clip_weights, clip_model, preprocess, device, hard_prompt, start, end):
    global train_loader_F_new
    global val_features_new, val_labels_new
    global test_features_new, test_labels_new

    print('\nLoading visual features and labels from new test set')
    cfg['subsample_classes'] = 'new'
    dataset_new = build_dataset(cfg, cfg['dataset'], cfg['root_path'], cfg['shots'])
    val_loader_new = build_data_loader(data_source=dataset_new.val, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)
    test_loader_new = build_data_loader(data_source=dataset_new.test, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)

    val_features_new, val_labels_new = pre_load_features(cfg, 'val', clip_model, val_loader_new, device)
    test_features_new, test_labels_new = pre_load_features(cfg, 'test', clip_model, test_loader_new, device)
    train_loader_F_new = build_data_loader(data_source=dataset_new.train_x, batch_size=64, tfm=train_transform, is_train=True, shuffle=True)
    clip_weights_new = clip_classifier(dataset_new.classnames, dataset_new.template, clip_model.float(), device)
    clip_weights_all = torch.cat((clip_weights, clip_weights_new), dim=1)

    global base_classnames, new_classnames, all_classnames
    base_classnames = dataset.classnames
    new_classnames = dataset_new.classnames
    all_classnames = base_classnames + new_classnames

    netE = Encoder().to(device)
    optimizer_e = torch.optim.SGD(netE.parameters(), lr=2e-3)
    scheduler_e = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_e, cfg['train_epoch'] * len(train_loader_F))

    netA = Encoder().to(device)
    optimizer_a = torch.optim.SGD(netA.parameters(), lr=2e-3)
    scheduler_a = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_a, cfg['train_epoch'] * len(train_loader_F))

    bbdm_model = Unet1D(device=device, dim = 256, dim_mults = (2,4), channels = 16)
    netB = GaussianDiffusion1D(bbdm_model, seq_length = 32, timesteps = 1000, sampling_timesteps=100, objective = 'ysubx').to(device)
    optimizer_b = torch.optim.Adam(netB.parameters(), lr=2e-5, betas = (0.9, 0.99))
    scheduler_b = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_b, cfg['train_epoch'] * len(train_loader_F))

    CrossEntropy = nn.CrossEntropyLoss()
    softmax = nn.Softmax()

    for train_idx in range(0, cfg['train_epoch'] + 1):
        netE.train()
        netA.train()
        netB.train()
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, cfg['train_epoch']))

        for i, (images, target) in enumerate(tqdm(train_loader_F)):
            images, target = images.to(device), target.to(device)
            with torch.no_grad():
                image_features = clip_model.encode_image(images).float()
                image_features /= image_features.norm(dim=-1, keepdim=True)

            netE.zero_grad()
            netA.zero_grad()
            netB.zero_grad()

            essential = netE(image_features) # bx512
            attribute = netA(image_features) # bx512

            b = image_features.size()[0]
            batch_image_features = image_features.view(b, 16, 32).float()
            batch_essential = essential.view(b, 16, 32).float()
            batch_attribute = attribute.view(b, 16, 32).float()
            loss_bbdm, _ = netB(batch_image_features, batch_essential, batch_attribute)

            logits_attribute = 100. * attribute.float() @ clip_weights_all.float()
            loss_ce = CrossEntropy(logits_attribute, target)
            logits_essential = 100. * essential.float() @ clip_weights_all.float()
            prediction = softmax(logits_essential)
            loss_kl = entropy(prediction)
            loss_all = loss_kl + loss_ce + loss_bbdm

            loss_all.backward()
            loss_list.append(loss_all.item())
            optimizer_e.step()
            scheduler_e.step()
            optimizer_a.step()
            scheduler_a.step()
            optimizer_b.step()
            scheduler_b.step()

        print('Loss: {:.4f}\n'.format(sum(loss_list)/len(loss_list)))

        if train_idx % 10 == 0:
            netE.eval()
            netA.eval()
            netB.eval()

            run_causal(cfg, netE, netA, clip_model, preprocess, device, hard_prompt)



def main():
    # Load config file
    torch.cuda.empty_cache()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    cfg = yaml.load(open('./configs/oxford_pets.yaml', 'r'), Loader=yaml.Loader)

    # Load cfg for conditional prompt.
    cfg['subsample_classes'] = "base"  # all, base or new

    print('Loading CLIP model.')
    device = 'cuda:0'
    clip_model, preprocess = clip.load('ViT-B-32.pt', device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    global train_loader_F
    global train_features, train_labels
    global val_features, val_labels
    global test_features, test_labels

    print("Preparing dataset.")
    dataset = build_dataset(cfg, cfg['dataset'], cfg['root_path'], cfg['shots'])

    val_loader = build_data_loader(data_source=dataset.val, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)
    test_loader = build_data_loader(data_source=dataset.test, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)
    train_loader_F = build_data_loader(data_source=dataset.train_x, batch_size=64, tfm=train_transform, is_train=True, shuffle=True)

    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    clip_weights = clip_classifier(dataset.classnames, dataset.template, clip_model) # 512x51

    print('\nLoading visual features and labels from val set')
    val_features, val_labels = pre_load_features(cfg, 'val', clip_model, val_loader, device)
    print('\nLoading visual features and labels from test set')
    test_features, test_labels = pre_load_features(cfg, 'test', clip_model, test_loader, device)
    print('\nLoading visual features and labels from train set')
    train_features, train_labels = pre_load_features(cfg, 'train', clip_model, train_loader_F, device)

    hard_prompt = []
    with torch.no_grad():
        for attribute in hard_attribute:
            texts = clip.tokenize(attribute, context_length=77).to(device)
            att_embeddings = clip_model.encode_text(texts)
            att_embeddings /= att_embeddings.norm(dim=-1, keepdim=True)
            att_embedding = att_embeddings.mean(dim=0)
            att_embedding /= att_embedding.norm()
            hard_prompt.append(att_embedding)
        hard_prompt = torch.stack(hard_prompt, dim=1).to(device)
        hard_prompt = hard_prompt.float().T

    run_bbdm(cfg, dataset, clip_weights, clip_model, preprocess, device, hard_prompt, start, end)


if __name__ == '__main__':
    main()