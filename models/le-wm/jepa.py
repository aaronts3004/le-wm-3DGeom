"""JEPA Implementation"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import torchvision.utils as vutils
import matplotlib.pyplot as plt

def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v

class JEPA(nn.Module):

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()

        self.sanity_checks = 0

    def encode(self, info):
        """Encode observations and actions into embeddings.
        info: dict with pixels and action keys
        """

        pixels = info['pixels'].float()

        # if self.sanity_checks == 0: 

        #     imgs = pixels[0]          # (T, C, H, W)

        #     mean = torch.tensor([0.485, 0.456, 0.406], device=imgs.device)[:, None, None]
        #     std  = torch.tensor([0.229, 0.224, 0.225], device=imgs.device)[:, None, None]

        #     ### RGB UNNORMALIZATION
        #     vis = imgs * std + mean
        #     vis = vis.clamp(0, 1)

        #     ### DEPTH UNNORMALIZATION
        #     # vis = (imgs + 1) / 2      # [-1,1] -> [0,1]

        #     vutils.save_image(
        #         vis,
        #         "check_encoder_input.png",
        #         nrow=imgs.shape[0],
        #     )

        #     self.sanity_checks += 1


            # first_frame = pixels[0, 0]      # B=0, T=0, C=0
            # plt.figure(figsize=(6,4))
            # plt.hist(first_frame.flatten().cpu().numpy(), bins=100)
            # plt.xlabel("Pixel value")
            # plt.ylabel("Count")
            # plt.title("Encoder input distribution (single frame)")
            # plt.tight_layout()
            # plt.savefig("depth_distribution.png", dpi=300)
            # plt.close()

            # plt.figure(figsize=(5,5))
            # plt.imshow(first_frame, cmap="viridis", vmin=-1, vmax=1)
            # plt.colorbar()
            # plt.savefig("first_frame_depth_map_cmap.png")
            # plt.close()

            # print("Encoder input:")
            # print(f"shape = {pixels.shape}")
            # print(f"min   = {pixels.min().item():.4f}")
            # print(f"max   = {pixels.max().item():.4f}")
            # print(f"mean  = {pixels.mean().item():.4f}")
            # print(f"std   = {pixels.std().item():.4f}")

            # num_pos = (pixels >= 0.999).float().mean()
            # num_neg = (pixels <= -0.999).float().mean()

            # print(f"% pixels at +1 : {100*num_pos:.2f}%")
            # print(f"% pixels at -1 : {100*num_neg:.2f}%")

            # print(torch.unique(first_frame).numel())
            # print(torch.quantile(first_frame.flatten(), torch.tensor([0.0,0.25,0.5,0.75,0.9,0.99,1.0])))


        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...") # flatten for encoding
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]  # cls token
        emb = self.projector(pixels_emb)
        
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def predict(self, emb, act_emb):
        """Predict next state embedding
        emb: (B, T, D)
        act_emb: (B, T, A_emb)
        """
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds

    ####################
    ## Inference only ##
    ####################

    def rollout(self, info, action_sequence, history_size: int = 3):
        """Rollout the model given an initial info dict and action sequence.
        pixels: (B, S, T, C, H, W)
        action_sequence: (B, S, T, action_dim)
         - S is the number of action plan samples                   # (by default 300)
         - T is the planning time horizon
        """

        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)
        
        # print("info['pixels'].shape=", info['pixels'].shape)
        # print("action_sequence.shape=", action_sequence.shape)


        B, S, T = action_sequence.shape[:3]
        # print(f"B={B},H={H},S={S},T={T}")


        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)         # split into 2 chunks, one of size (H) one of size(T-H)
        info["action"] = act_0
        n_steps = T - H

        # print("n_steps=", n_steps)

        # copy and encode initial info dict
        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init)
        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)
        _init = {k: detach_clone(v) for k, v in _init.items()}

        # flatten batch and sample dimensions for rollout
        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        # print("info['action'].shape =", info["action"].shape)
        # print("info['goal'].shape   =", info["goal"].shape)



        # rollout predictor autoregressively for n_steps
        HS = history_size
        # print("HS =", HS)
        # print("Initial emb.shape =", emb.shape)
        for t in range(n_steps):

            act_emb = self.action_encoder(act)
            emb_trunc = emb[:, -HS:]                # (BS, HS, D)
            act_trunc = act_emb[:, -HS:]            # (BS, HS, A_emb)
            pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]       # (BS, 1, D)
            emb = torch.cat([emb, pred_emb], dim=1)                     # (BS, T+1, D)

            next_act = act_future[:, t : t + 1, :]  # (BS, 1, action_dim)
            act = torch.cat([act, next_act], dim=1)  # (BS, T+1, action_dim)


        # predict the last state
        act_emb = self.action_encoder(act)  # (BS, T, A_emb)
        emb_trunc = emb[:, -HS:]  # (BS, HS, D)
        act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
        pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
        emb = torch.cat([emb, pred_emb], dim=1)

        # unflatten batch and sample dimensions
        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout

        return info

    def criterion(self, info_dict: dict):
        """Compute the cost between predicted embeddings and goal embeddings."""
        pred_emb = info_dict["predicted_emb"]  # (B,S, T-1, dim)
        goal_emb = info_dict["goal_emb"]  # (B, S, T, dim)

        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)

        # return last-step cost per action candidate
        cost = F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction="none",
        ).sum(dim=tuple(range(2, pred_emb.ndim)))  # (B, S)

        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """ Compute the cost of action candidates given an info dict with goal and initial state."""

        assert "goal" in info_dict, "goal not in info_dict"

        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]

        for k in info_dict:
            if k.startswith("goal_"):
                goal[k[len("goal_") :]] = goal.pop(k)

        goal.pop("action")
        goal = self.encode(goal)

        info_dict["goal_emb"] = goal["emb"]
        info_dict = self.rollout(info_dict, action_candidates)

        cost = self.criterion(info_dict)
        
        return cost
