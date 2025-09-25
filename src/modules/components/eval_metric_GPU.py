import gc
import numpy as np
import torch
from functools import partial
from torchvision.transforms import GaussianBlur
from src.modules.components.insertion_deletion import BaseEvaluation
from src.modules.components.correlation import correlation_functions
from src.modules.components.perturb import perturb_methods
from src.modules.components.perturbation_delta import perturbation_effects
from src.modules.components.preservation import preservation_effect


class Insertion_GPU(BaseEvaluation):
    def __init__(self, pixel_batch_size=10, sigma=5.0, kernel_size=9, modality="image",perturbation='replace',preserve='target'):
        self.sigma = sigma
        self.pixel_batch_size = pixel_batch_size
        if kernel_size == 0 and sigma == 0:
            self.gaussian_blurr = torch.zeros_like
        else:
            self.gaussian_blurr = GaussianBlur(kernel_size, sigma)
        self.modality = modality
        self.perturb_func = perturb_methods[perturbation]
        self.perservation = preservation_effect[preserve]

    @torch.no_grad()
    def __call__(self, model, x_batch, y_batch, a_batch, **kwargs):
        
        device = next(model.parameters()).device

        # x_batch = torch.tensor(x_batch).to(device)
        # a_batch = torch.tensor(a_batch).to(device)
        y_origin = torch.softmax(model(x_batch),dim=1)

        # baseline input
        blurred_input = self.gaussian_blurr(x_batch)

        self.classifier = model
        self.classifier.eval()

        num_pixels = torch.numel(a_batch[0])
        steps = range(0, num_pixels, self.pixel_batch_size)
        # print(steps)
        probs = []

        _, all_indices = torch.topk(a_batch.flatten(start_dim=1), num_pixels, dim=1)

        x_batch_flatten = x_batch.flatten(start_dim=1).clone().to(device)
        blurred_input_flatten = blurred_input.flatten(start_dim=1).clone().to(device)

        for i in steps:
            k = min(i + self.pixel_batch_size, num_pixels)
  
            indices = all_indices[:, :k]

            x_batch_modified = self.perturb_func(src=x_batch_flatten,trg=blurred_input_flatten,indices=indices)
            
            x_batch_modified = x_batch_modified.reshape(x_batch.shape)
            logits = self.classifier(x_batch_modified)
            preds = torch.softmax(logits, dim=-1)
            # batch_indices = torch.arange(preds.shape[0], device=device)
            # y_pred = preds[batch_indices, y_batch_f]

            y_pred = self.perservation(y_origin,preds)
            probs.append(y_pred)
            del x_batch_modified, logits, preds, y_pred
            torch.cuda.empty_cache()
            # if (i/self.pixel_batch_size)%50 == 0:
            #     print(probs[-1])

        insertion_curve = torch.stack(probs,dim=-1).to(device)
        scores = torch.trapz(insertion_curve,torch.arange(0, num_pixels, self.pixel_batch_size).to(device)/num_pixels,dim=-1)
        # print(insertion_curve,scores)
        return scores

class Deletion_GPU(BaseEvaluation):
    def __init__(self, pixel_batch_size=10, modality="image",perturbation='replace',preserve='target'):
        self.pixel_batch_size = pixel_batch_size
        self.modality = modality
        self.perturb_func = perturb_methods[perturbation]
        self.perservation = preservation_effect[preserve]

    @torch.no_grad()
    def __call__(self, model, x_batch, y_batch, a_batch, **kwargs):
        
        device = next(model.parameters()).device

        self.classifier = model
        self.classifier.eval()
        # x_batch = torch.tensor(x_batch).to(device)
        # a_batch = torch.tensor(a_batch).to(device)
        y_origin = torch.softmax(model(x_batch),dim=1)

        num_pixels = torch.numel(a_batch[0])
        steps = range(0, num_pixels, self.pixel_batch_size)
        # print(steps)
        probs = []

        _, all_indices = torch.topk(a_batch.flatten(start_dim=1), num_pixels, dim=1)
        
        x_batch_flatten = x_batch.flatten(start_dim=1).clone().to(device)

        for i in steps:
            k = min(i + self.pixel_batch_size, num_pixels)
     
            indices = all_indices[:, :k]
            x_batch_flatten = self.perturb_func(src=torch.zeros_like(x_batch_flatten),trg=x_batch_flatten,indices=indices)
            x_batch_modified = x_batch_flatten.reshape(x_batch.shape)
            logits = self.classifier(x_batch_modified)
            preds = torch.softmax(logits, dim=-1)

            y_pred = self.perservation(y_origin,preds)
            # print(y_pred)
            probs.append(y_pred)
            del x_batch_modified, logits, preds, y_pred
            torch.cuda.empty_cache()
            # print(y_pred)
        deletion_curve = torch.stack(probs,dim=-1).to(device)
        scores = torch.trapz(deletion_curve,torch.arange(0, num_pixels, self.pixel_batch_size).to(device)/num_pixels,dim=-1)
        # print(scores,deletion_curve)
        del x_batch_flatten, all_indices, probs, steps, y_origin
        gc.collect()
        torch.cuda.empty_cache()
        return scores

class NegativePerturbation_GPU(BaseEvaluation):
    def __init__(self, pixel_batch_size=10, modality="image",perturbation='replace',preserve='target'):
        self.pixel_batch_size = pixel_batch_size
        self.modality = modality
        self.perturb_func = perturb_methods[perturbation]
        self.perservation = preservation_effect[preserve]

    @torch.no_grad()
    def __call__(self, model, x_batch, y_batch, a_batch, **kwargs):
        
        device = next(model.parameters()).device
        self.classifier = model
        self.classifier.eval()
        # x_batch = torch.tensor(x_batch).to(device)
        # a_batch = torch.tensor(a_batch).to(device)
        y_origin = torch.softmax(model(x_batch),dim=1)

        # print(y_origin.shape)
        # print(x_batch_f.shape)
        num_pixels = torch.numel(a_batch[0])
        steps = range(0, num_pixels, self.pixel_batch_size)
        # print(steps)
        probs = []

        _, all_indices = torch.topk(a_batch.flatten(start_dim=1), num_pixels, dim=1)

        all_indices = torch.flip(all_indices, dims=[1])
  
        x_batch_flatten = x_batch.flatten(start_dim=1).clone().to(device)

        t = torch.full((a_batch.shape[0],), -1, device=device)
    
        frozen_probs = torch.zeros(a_batch.shape[0], device=device)

        for i in steps:
            k = min(i + self.pixel_batch_size, num_pixels)

            indices = all_indices[:, :k]

            x_batch_flatten = self.perturb_func(src=torch.zeros_like(x_batch_flatten),trg=x_batch_flatten,indices=indices)
            x_batch_modified = x_batch_flatten.reshape(x_batch.shape)
            logits = self.classifier(x_batch_modified)
            preds = torch.softmax(logits, dim=-1)
            target_pred = torch.argmax(preds, dim=-1)

            # batch_indices = torch.arange(preds.shape[0], device=preds.device)
            # current_y_pred = preds[batch_indices, y_batch_f]
            current_y_pred = self.perservation(y_origin,preds)
       
            new_changed_mask = (target_pred != y_origin.argmax(dim=1)) & (t == -1)
            t[new_changed_mask] = i
            frozen_probs[new_changed_mask] = current_y_pred[new_changed_mask]

        
            y_pred = torch.where(t != -1, frozen_probs, current_y_pred)
            probs.append(y_pred)
            del x_batch_modified, logits, preds, y_pred, target_pred, current_y_pred, new_changed_mask
            torch.cuda.empty_cache()
        deletion_curve = torch.stack(probs,dim=-1).to(device)
        scores = torch.trapz(deletion_curve,torch.arange(0, num_pixels, self.pixel_batch_size).to(device)/num_pixels,dim=-1)
        # print(scores,deletion_curve)
        del x_batch_flatten, all_indices, probs, steps, y_origin, t, frozen_probs
        gc.collect()
        torch.cuda.empty_cache()
        return scores

class PositivePerturbation_GPU(BaseEvaluation):
    def __init__(self, pixel_batch_size=10, modality="image",perturbation='replace',preserve='target'):
        self.pixel_batch_size = pixel_batch_size
        self.modality = modality
        self.perturb_func = perturb_methods[perturbation]
        self.perservation = preservation_effect[preserve]

    @torch.no_grad()
    def __call__(self, model, x_batch, y_batch, a_batch, **kwargs):
        
        self.classifier = model
        self.classifier.eval()
        device = next(model.parameters()).device

        # x_batch = torch.tensor(x_batch).to(device)
        # a_batch = torch.tensor(a_batch).to(device)
        y_origin = torch.softmax(model(x_batch),dim=1)

        num_pixels = torch.numel(a_batch[0])
        steps = range(0, num_pixels, self.pixel_batch_size)
        # print(steps)
        probs = []

        _, all_indices = torch.topk(a_batch.flatten(start_dim=1), num_pixels, dim=1)

        x_batch_flatten = x_batch.flatten(start_dim=1).clone().to(device)

        t = torch.full((a_batch.shape[0],), -1, device=device)

        frozen_probs = torch.zeros(a_batch.shape[0], device=device)

        for i in steps:
            k = min(i + self.pixel_batch_size, num_pixels)
       
            indices = all_indices[:, :k]

            x_batch_flatten = self.perturb_func(src=torch.zeros_like(x_batch_flatten),trg=x_batch_flatten,indices=indices)
            x_batch_modified = x_batch_flatten.reshape(x_batch.shape)
            logits = self.classifier(x_batch_modified)
            preds = torch.softmax(logits, dim=-1)
            target_pred = torch.argmax(preds, dim=-1)

            # batch_indices = torch.arange(preds.shape[0], device=preds.device)
            # current_y_pred = preds[batch_indices, y_batch_f]
            current_y_pred = self.perservation(y_origin,preds)

            new_changed_mask = (target_pred != y_origin.argmax(dim=1)) & (t == -1)
            t[new_changed_mask] = i
            frozen_probs[new_changed_mask] = current_y_pred[new_changed_mask]

            y_pred = torch.where(t != -1, frozen_probs, current_y_pred)
            probs.append(y_pred)
            del x_batch_modified, logits, preds, y_pred, target_pred, current_y_pred, new_changed_mask
            torch.cuda.empty_cache()
        deletion_curve = torch.stack(probs,dim=-1).to(device)
        scores = torch.trapz(deletion_curve,torch.arange(0, num_pixels, self.pixel_batch_size).to(device)/num_pixels,dim=-1)
        # print(scores,deletion_curve)
        del x_batch_flatten, all_indices, probs, steps, y_origin, t, frozen_probs
        gc.collect()
        torch.cuda.empty_cache()
        return scores

class FaithfulnessCorrelation_GPU(BaseEvaluation):
    def __init__(
        self,
        nr_runs=100,
        subset_size=224,
        similarity_func='pearson',
        perturb_effect='minus',
        perturbation = 'replace'
    ):
        self.nr_runs = nr_runs
        self.subset_size = subset_size
        self.perturb_func = perturb_methods[perturbation]
        self.similarity_func = correlation_functions[similarity_func]
        self.perturb_effect = perturbation_effects[perturb_effect]

    @torch.no_grad()
    def __call__(self, model, x_batch, y_batch, a_batch, **kwargs):
        device = next(model.parameters()).device
        # x_batch = torch.tensor(x_batch).to(device)
        # a_batch = torch.tensor(a_batch).to(device)
        y_origin = torch.softmax(model(x_batch),dim=1)
        # y_batch = torch.argmax(y_origin,dim=1)
        # print(x_batch.shape,y_batch.shape,a_batch)
        batch_size = x_batch.shape[0]
        num_features = a_batch[0].numel()
        a_batch_flat = a_batch.reshape(batch_size, -1)
        x_batch_flat = x_batch.reshape(batch_size, -1)

        pred_deltas = []
        att_sums = []

        for _ in range(self.nr_runs):
            indices = torch.stack([
                torch.randperm(num_features, device=device)[:self.subset_size]
                # torch.randperm(num_features, device=device)[:subset_size]
                for _ in range(batch_size)
            ])
            x_perturbed_flat = x_batch_flat.clone()

            x_perturbed = self.perturb_func(src=torch.zeros_like(x_perturbed_flat),trg=x_perturbed_flat,indices=indices).reshape(x_batch.shape)
 
            logits_perturb = model(x_perturbed)
            preds_perturb = torch.softmax(logits_perturb, dim=-1)
            
            pred_deltas.append(self.perturb_effect(y_origin,preds_perturb))
   
            att_sum = a_batch_flat.gather(1, indices).sum(dim=1)
            att_sums.append(att_sum)
            del x_perturbed_flat

        pred_deltas = torch.stack(pred_deltas, dim=1) # [batch, nr_runs]
        att_sums = torch.stack(att_sums, dim=1)        # [batch, nr_runs]

        # scores = []
        corr = self.similarity_func(att_sums,pred_deltas)

        return corr

class FaithfulnessEstimate_GPU(BaseEvaluation):
    def __init__(
        self,
        features_in_step=100,
        perturbation='replace',
        perturb_effect='minus',
        similarity_func='pearson'
    ):
        self.features_in_step = features_in_step
        self.similarity_func = correlation_functions[similarity_func]
        self.perturb_func = perturb_methods[perturbation]
        self.perturb_effect = perturbation_effects[perturb_effect]

    @torch.no_grad()
    def __call__(self, model, x_batch, y_batch, a_batch, **kwargs):
        device = next(model.parameters()).device
        model.eval()

        # x_batch = torch.tensor(x_batch).to(device)
        # a_batch = torch.tensor(a_batch).to(device)
        y_origin = torch.softmax(model(x_batch),dim=1)

        batch_size = x_batch.shape[0]
        num_features = a_batch[0].numel()

        a_batch_flat = a_batch.reshape(batch_size, -1)
        x_batch_flat = x_batch.reshape(batch_size, -1)

        a_indices = torch.argsort(-a_batch_flat, dim=1)

        n_perturbations = torch.ceil(torch.tensor(num_features / self.features_in_step)).int()
        pred_deltas = []
        att_sums = []
        for perturbation_step_index in range(n_perturbations):

            start = perturbation_step_index * self.features_in_step
            end = min((perturbation_step_index + 1) * self.features_in_step, num_features)
            indices = a_indices[:, start:end]

            x_perturbed_flat = x_batch_flat.clone()
            x_perturbed = self.perturb_func(
                src=torch.zeros_like(x_perturbed_flat),
                trg=x_perturbed_flat,
                indices=indices).reshape(x_batch.shape)

            logits_perturb = model(x_perturbed)
            preds_perturb = torch.softmax(logits_perturb, dim=-1)

            pred_deltas.append(self.perturb_effect(y_origin,preds_perturb))
     
            att_sum = a_batch_flat.gather(1, indices).sum(dim=1)
            att_sums.append(att_sum)
        pred_deltas = torch.stack(pred_deltas, dim=1)  # [batch, n_perturbations]
        att_sums = torch.stack(att_sums, dim=1)        # [batch, n_perturbations]

        corr = self.similarity_func(att_sums,pred_deltas)

        return corr

class MonotonicityCorrelation_GPU(BaseEvaluation):
    def __init__(
        self,
        similarity_func='pearson',
        features_in_step=100,
        perturbation='replace',
        perturb_effect='variance'
    ):
        self.similarity_func = correlation_functions[similarity_func]
        self.features_in_step = features_in_step
        self.perturb_func = perturb_methods[perturbation]
        self.perturb_effect = perturbation_effects[perturb_effect]


    @torch.no_grad()
    def __call__(self, model, x_batch, y_batch, a_batch, **kwargs):
        device = next(model.parameters()).device
        model.eval()
        # x_batch = torch.tensor(x_batch).to(device)
        # a_batch = torch.tensor(a_batch).to(device)
        y_origin = torch.softmax(model(x_batch),dim=1)
        #y_batch = torch.argmax(y_origin,dim=1)

        batch_size = x_batch.shape[0]
        num_features = a_batch[0].numel()
        a_batch_flat = a_batch.reshape(batch_size, -1)
        x_batch_flat = x_batch.reshape(batch_size, -1)

        a_indices = torch.argsort(a_batch_flat, dim=1)

        n_perturbations = torch.ceil(torch.tensor(num_features / self.features_in_step)).int()
        atts = []
        vars = []
        for perturbation_step_index in range(n_perturbations):
   
            start = perturbation_step_index * self.features_in_step
            end = min((perturbation_step_index + 1) * self.features_in_step, num_features)
            a_ix = a_indices[:, start:end]
      
            x_perturbed_flat = x_batch_flat.clone()
            x_perturbed = self.perturb_func(src=torch.zeros_like(x_perturbed_flat),trg=x_perturbed_flat,indices=a_ix).reshape(x_batch.shape)

            logits_perturb = model(x_perturbed)
            preds_perturb = torch.softmax(logits_perturb, dim=-1)

            var = self.perturb_effect(y_origin,preds_perturb) # B

            vars.append(var)
            att_sum = a_batch_flat.gather(1, a_ix).sum(dim=1)
            atts.append(att_sum)
            # print(vars[-1],atts[-1])

        vars = torch.stack(vars, dim=1)  # B*N
        atts = torch.stack(atts, dim=1)  # B*N
        # print(vars.shape,atts.shape)

        similarities = self.similarity_func(vars,atts)
        # print(similarities)
        del x_batch, a_batch, y_origin, batch_size, num_features, a_batch_flat, x_batch_flat, a_indices, n_perturbations, atts, vars, a_ix, x_perturbed_flat, x_perturbed, logits_perturb, preds_perturb
        gc.collect()
        torch.cuda.empty_cache()
        return similarities

class IROF_GPU(BaseEvaluation):
    def __init__(
        self,
        pixel_batch_size=100,
        perturbation='replace',
        preserve='ratio'
    ):
        self.pixel_batch_size = pixel_batch_size
        self.perturb_func = perturb_methods[perturbation]
        self.perservation = preservation_effect[preserve]

    @torch.no_grad()
    def __call__(self, model, x_batch, y_batch, a_batch, **kwargs):
        device = next(model.parameters()).device
        x_batch = torch.as_tensor(x_batch, device=device)
        a_batch = torch.as_tensor(a_batch, device=device)
        y_origin = torch.softmax(model(x_batch),dim=1)

        a_batch_f = a_batch.flatten(start_dim=1).clone().to(device)

        num_pixels = torch.numel(a_batch_f[0])
        steps = range(0, num_pixels, self.pixel_batch_size)

        probs = []

        _, all_indices = torch.topk(a_batch_f.flatten(start_dim=1), num_pixels, dim=1)

        x_batch_flatten = x_batch.flatten(start_dim=1).clone().to(device)

        for i in steps:
            k = min(i + self.pixel_batch_size, num_pixels)
         
            indices = all_indices[:, :k]

            x_batch_modified = self.perturb_func(src=torch.zeros_like(x_batch_flatten),trg=x_batch_flatten,indices=indices)
            x_batch_modified = x_batch_modified.reshape(x_batch.shape)

            logits = model(x_batch_modified)
            preds = torch.softmax(logits, dim=-1)

            y_pred = self.perservation(y_origin,preds)
            probs.append(1 - y_pred)
            del x_batch_modified, logits, preds, y_pred
            torch.cuda.empty_cache()

            # if (i/self.pixel_batch_size)%50 == 0:
            #     print(probs[-1])

        aoc_curve = torch.stack(probs,dim=-1).to(device)
        scores = torch.trapz(aoc_curve,torch.arange(0, num_pixels, self.pixel_batch_size).to(device)/num_pixels,dim=-1)
        # print(aoc_curve,scores)
        del x_batch, a_batch, y_origin, a_batch_f, x_batch_flatten, all_indices,  probs, steps
        gc.collect()
        torch.cuda.empty_cache()
        return scores

class RegionPerturbation_GPU(BaseEvaluation):
    def __init__(
        self,
        regions_evaluation=100,
        perturbation='replace',
        perturb_effect='minus'
    ):
        self.patch_size = regions_evaluation
        self.perturb_func = perturb_methods[perturbation]
        self.perturb_effect = perturbation_effects[perturb_effect]

    @torch.no_grad()
    def __call__(self, model, x_batch, y_batch, a_batch, **kwargs):
        device = next(model.parameters()).device
        
        self.classifier = model
        self.classifier.eval()
        # x_batch = torch.tensor(x_batch).to(device)
        # a_batch = torch.tensor(a_batch).to(device)
        y_origin = torch.softmax(model(x_batch),dim=1)

        num_pixels = torch.numel(a_batch[0])
        steps = range(0, num_pixels, self.patch_size)
        # print(steps)
        probs = []

        _, all_indices = torch.topk(a_batch.flatten(start_dim=1), num_pixels, dim=1)
        
        x_batch_flatten = x_batch.flatten(start_dim=1).clone().to(device)

        for i in steps:
            k = min(i + self.patch_size, num_pixels)
  
            indices = all_indices[:, :k]
     
            x_batch_flatten = self.perturb_func(src=torch.zeros_like(x_batch_flatten),trg=x_batch_flatten,indices=indices)
            x_batch_modified = x_batch_flatten.reshape(x_batch.shape)
            logits = self.classifier(x_batch_modified)
            preds = torch.softmax(logits, dim=-1)

            y_pred = self.perturb_effect(y_origin,preds)

            probs.append(y_pred)
            del x_batch_modified, logits, preds, y_pred
            torch.cuda.empty_cache()

        effects = torch.stack(probs,dim=1)
        scores = torch.sum(effects,dim=1)/len(steps)
        # print(effects,scores)
        del x_batch_flatten, all_indices, probs, steps, y_origin, effects
        gc.collect()
        torch.cuda.empty_cache()
        return scores

class Infidelity_GPU(BaseEvaluation):
    def __init__(
        self,
        nr_runs=100,
        similarity_func='pearson',
        perturb_effect='minus',
        perturbation = 'replace'
    ):
        self.nr_runs = nr_runs
        self.perturb_func = perturb_methods[perturbation]
        self.similarity_func = correlation_functions[similarity_func]
        self.perturb_effect = perturbation_effects[perturb_effect]

    @torch.no_grad()
    def __call__(self, model, x_batch, y_batch, a_batch, **kwargs):
        device = next(model.parameters()).device

        y_origin = torch.softmax(model(x_batch),dim=1)
 

        batch_size = x_batch.shape[0]
        num_features = a_batch[0].numel()
        a_batch_flat = a_batch.reshape(batch_size, -1)
        x_batch_flat = x_batch.reshape(batch_size, -1)

        pred_deltas = []
        att_sums = []

        for _ in range(self.nr_runs):
            bernoulli_mask = torch.bernoulli(torch.full((num_features,), 0.5, device=device))  # shape: (N,)
            bernoulli_mask = bernoulli_mask.unsqueeze(0).repeat(batch_size, 1)  # shape: (B, N)
            
          
            indices_1d = torch.nonzero(bernoulli_mask[0], as_tuple=False).squeeze(1)  # shape: (k,)
            indices = indices_1d.unsqueeze(0).repeat(batch_size, 1)  # shape: (B, k)

 
            x_perturbed_flat = x_batch_flat.clone()

            x_perturbed = self.perturb_func(src=torch.zeros_like(x_perturbed_flat),trg=x_perturbed_flat,indices=indices).reshape(x_batch.shape)
     
            logits_perturb = model(x_perturbed)
            preds_perturb = torch.softmax(logits_perturb, dim=-1)
            # y_pred_perturb = preds_perturb[batch_indices, y_batch]
            
            pred_deltas.append(self.perturb_effect(y_origin,preds_perturb))

            att_sum = a_batch_flat.gather(1, indices).sum(dim=1)
            att_sums.append(att_sum)

        pred_deltas = torch.stack(pred_deltas, dim=1) # [batch, nr_runs]
        att_sums = torch.stack(att_sums, dim=1)        # [batch, nr_runs]

        corr = self.similarity_func(att_sums,pred_deltas)
        # print(corr)
        del x_batch, a_batch, y_origin, batch_size, num_features, a_batch_flat, x_batch_flat, pred_deltas, att_sums, bernoulli_mask, indices_1d, indices, x_perturbed_flat, x_perturbed, logits_perturb, preds_perturb
        gc.collect()
        torch.cuda.empty_cache()
        return corr

