import numpy as np

import torch
import torch.nn as nn

from models.model_utils import SNN_Block, BilinearFusion
from models.nystromformer_encoder import Transformer_Encoder_Cls

import warnings

warnings.filterwarnings("ignore")


class GMSL(nn.Module):
    def __init__(self,
                 omic_sizes=[100, 200, 300, 400, 500, 600, 700, 800],
                 n_classes=4,
                 fusion="concat",
                 model_size="small",
                 mask_ratio=0.75,
                 decoder_embed_dim=128,
                 dropout=0.25):
        super(GMSL, self).__init__()

        self.omic_sizes = omic_sizes
        self.n_classes = n_classes
        self.fusion = fusion
        self.mask_ratio = mask_ratio

        # pathology modified by pt_file
        self.size_dict = {"path": {"small": [1024, 256, 256],   
                                   # "large": [1024, 512, 256],
                                   "trans": [768, 256, 256]},
                          "omics": {"small": [256, 256],
                                    # "large": [1024, 1024, 1024, 256],
                                    "trans": [256, 256]}}  # [1024, 1024, 1024, 256], BRCA:[1024, 256]

        # ====== Pathology Embedding ======
        wsi_hidden = self.size_dict["path"][model_size]
        fc = []
        for idx in range(len(wsi_hidden) - 1):
            fc.append(nn.Linear(wsi_hidden[idx], wsi_hidden[idx + 1]))
            fc.append(nn.ReLU())
            fc.append(nn.Dropout(0.25))
        self.pathology_fc = nn.Sequential(*fc)  # 2 layers

        # ====== Genomic Embedding, SNN ======
        snn_hidden = self.size_dict["omics"][model_size]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=snn_hidden[0])]
            for i, _ in enumerate(snn_hidden[1:]):
                fc_omic.append(SNN_Block(dim1=snn_hidden[i], dim2=snn_hidden[i + 1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.genomics_fc = nn.ModuleList(sig_networks)  # 2 layers

        # ====== Encoder and Decoder Transformer ======
        self.pathology_encoder = Transformer_Encoder_Cls(dim=wsi_hidden[-1], num_layers=2)
        self.genomics_encoder = Transformer_Encoder_Cls(dim=snn_hidden[-1])
        self.pathology_decoder = Transformer_Encoder_Cls(dim=wsi_hidden[-1], num_layers=2)
        self.genomics_decoder = Transformer_Encoder_Cls(dim=snn_hidden[-1])

        # ====== Cross-omics Attention ======
        self.P_in_G_Att = nn.MultiheadAttention(embed_dim=256, num_heads=1)  # P->G Attention
        self.G_in_P_Att = nn.MultiheadAttention(embed_dim=256, num_heads=1)  # G->P Attention

        # ====== Pretext task ======
        # decoder
        self.decoder_embed = nn.Linear(wsi_hidden[-1], decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.pathology_reconstruction_decoder = Transformer_Encoder_Cls(dim=decoder_embed_dim,
                                                                       num_layers=2,
                                                                       add_cls=False)
        self.decoder_pred = nn.Sequential(*[nn.Linear(decoder_embed_dim, wsi_hidden[-1]),
                                            nn.ReLU(),
                                            nn.Linear(wsi_hidden[-1], wsi_hidden[0])])

        # ====== Fusion Layer ======
        if self.fusion == "concat":
            self.mm = nn.Sequential(*[nn.Linear(wsi_hidden[-1] * 4, wsi_hidden[-1]),
                                      nn.ReLU(),
                                      nn.Linear(wsi_hidden[-1], wsi_hidden[-1]),
                                      nn.ReLU()])  # 2 layers
        elif self.fusion == "bilinear":
            self.mm = BilinearFusion(dim1=wsi_hidden[-1], dim2=wsi_hidden[-1],
                                     scale_dim1=8, scale_dim2=8,
                                     mmhid=wsi_hidden[-1])
        else:
            raise NotImplementedError("Fusion [{}] is not implemented".format(self.fusion))

        # ====== Survival Layer ======
        self.classifier = nn.Linear(wsi_hidden[-1], self.n_classes)

    # ====== pretext task ======
    def random_masking(self, x, mask_ratio):
        B, L, D = x.shape 
        len_keep = int(L * (1 - mask_ratio))  

        # generate random noise
        noise = torch.rand(B, L, device=x.device)  
        ids_shuffle = torch.argsort(noise, dim=0)  
        ids_restore = torch.argsort(ids_shuffle, dim=1)  

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]  # small is keep, large is remove
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))  

        # generate the binary mask
        mask = torch.ones([B, L], device=x.device)
        mask[:, :len_keep] = 0  # 0 is keep, 1 is remove
        mask = torch.gather(mask, dim=1, index=ids_restore)  # ids_restore restore mask matrix

        return x_masked, mask, ids_restore

    def forward_encoder(self, x_path, x_omic, mask_ratio):
        """
            x_path: [1,N,1024]
            x_omic: 8 * [l] list
        """
        x_path_masked, mask, ids_restore = self.random_masking(x_path.unsqueeze(0), mask_ratio)

        # ====== WSI, Pathology FC ======
        pathology_features = self.pathology_fc(x_path_masked)  

        # ====== Genomics embedding, SNN ======
        genomics_features = [self.genomics_fc[idx].forward(sig_feature) for idx, sig_feature in enumerate(x_omic)]
        genomics_features = torch.stack(genomics_features).unsqueeze(0) 

        # ====== Encoder ======
        # patch for Cross Attention; cls for classification
        cls_token_pathology_encoder, patch_token_pathology_encoder = \
            self.pathology_encoder(pathology_features)  
        cls_token_genomics_encoder, patch_token_genomics_encoder = \
            self.genomics_encoder(genomics_features)  

        # ====== Cross-omics attention, P in G ======
        latent, _ = \
            self.P_in_G_Att(patch_token_pathology_encoder.transpose(1, 0),
                            patch_token_genomics_encoder.transpose(1, 0),
                            patch_token_genomics_encoder.transpose(1, 0))  


        return latent, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x.transpose(1, 0))  

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)  
        x = torch.cat([x, mask_tokens], dim=1)  
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        # add position embedding, but cannot
        x = self.pathology_reconstruction_decoder(x)  
        x = self.decoder_pred(x)  

        return x

    def forward_loss(self, x, pred, mask):
        """
        imgs: [N,1024]
        pred: [1, N, 1024]
        mask: [b, N], 0 is keep, 1 is remove,
        """

        loss = (pred - x.unsqueeze(0)) ** 2  
        loss = loss.mean(dim=-1)  #  mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, x_path, x_omic, mode='finetune'):

        if mode == 'pretrain':
            # ============ pretext task ============
            latent, mask, ids_restore = self.forward_encoder(x_path, x_omic, mask_ratio=self.mask_ratio)
            reconstruction = self.forward_decoder(latent, ids_restore)
            loss = self.forward_loss(x_path, reconstruction, mask)

            return loss

        elif mode == 'finetune':
            # ====== WSI, Pathology FC ======
            pathology_features = self.pathology_fc(x_path).unsqueeze(0) 

            # ====== Genomics embedding, SNN ======
            genomics_features = [self.genomics_fc[idx].forward(sig_feature) for idx, sig_feature in enumerate(x_omic)]
            genomics_features = torch.stack(genomics_features).unsqueeze(0)  

            # ====== Encoder ======
            # patch for cross attention, cls for survival prediction
            cls_token_pathology_encoder, patch_token_pathology_encoder = \
                self.pathology_encoder(pathology_features)  
            cls_token_genomics_encoder, patch_token_genomics_encoder = \
                self.genomics_encoder(genomics_features)  

            # ====== Cross-omics attention ======
            pathomics_in_genomics, _ = \
                self.P_in_G_Att(patch_token_pathology_encoder.transpose(1, 0),
                                patch_token_genomics_encoder.transpose(1, 0),
                                patch_token_genomics_encoder.transpose(1, 0))  
            genomics_in_pathomics, _ = \
                self.G_in_P_Att(patch_token_genomics_encoder.transpose(1, 0),
                                patch_token_pathology_encoder.transpose(1, 0),
                                patch_token_pathology_encoder.transpose(1, 0))  

            # ====== Decoder =====
            # pathology decoder: only use cls_token for now
            cls_token_pathology_decoder, _ = self.pathology_decoder(
                pathomics_in_genomics.transpose(1, 0))  
            # genomics decoder
            cls_token_genomics_decoder, _ = self.genomics_decoder(genomics_in_pathomics.transpose(1, 0))  

            # ======== Late fusion ==========
            if self.fusion == "concat":
                fusion = self.mm(torch.concat((cls_token_pathology_encoder, cls_token_pathology_decoder,
                                               cls_token_genomics_encoder, cls_token_genomics_decoder),
                                              dim=1))  # take cls token to make prediction

            elif self.fusion == "bilinear":
                fusion = self.mm(
                    (cls_token_pathology_encoder + cls_token_pathology_decoder) / 2,
                    (cls_token_genomics_encoder + cls_token_genomics_decoder) / 2,
                )  # take cls token to make prediction

            else:
                raise NotImplementedError("Fusion [{}] is not implemented".format(self.fusion))

            # ====== Survival Layer ======
            logits = self.classifier(fusion)  
            hazards = torch.sigmoid(logits)  
            S = torch.cumprod(1 - hazards, dim=1)  

            return hazards, S


def main():
    # add .unsqueeze(0) to match dataloader format
    x_path = torch.tensor(np.zeros((1129, 1024), dtype=np.float32)).unsqueeze(0)

    x_omics = [torch.tensor(np.zeros(352, dtype=np.float32)).unsqueeze(0),
               torch.tensor(np.zeros(440, dtype=np.float32)).unsqueeze(0),
               torch.tensor(np.zeros(243, dtype=np.float32)).unsqueeze(0),
               torch.tensor(np.zeros(314, dtype=np.float32)).unsqueeze(0),
               torch.tensor(np.zeros(501, dtype=np.float32)).unsqueeze(0),
               torch.tensor(np.zeros(1462, dtype=np.float32)).unsqueeze(0),
               torch.tensor(np.zeros(276, dtype=np.float32)).unsqueeze(0),
               torch.tensor(np.zeros(82, dtype=np.float32)).unsqueeze(0)]

    # Load models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GMSL(omic_sizes=[i.shape[1] for i in x_omics],
                 n_classes=4,
                 fusion='concat',
                 mask_ratio=0.75)
    model.to(device)

    data_WSI = x_path.squeeze(0).to(device)
    data_omic = [i.squeeze(0).to(device) for i in x_omics]

    print(f"data_WSI.shape: {data_WSI.shape}")
    print(f"data_omic.shape: {[i.shape for i in data_omic]}\n\n")

    loss = model(data_WSI, data_omic, mode='pretrain')
    print(f"loss: {loss}")
    hazards, Survival = model(x_path=data_WSI, x_omic=data_omic, mode='finetune')
    print(f"hazards: {hazards}, Survival: {Survival}")

    # # Pretrain-path FLOPS
    # flops, params = profile(models, inputs=(data_WSI, data_omic, 'pretrain-pathology'))
    # flops, params = clever_format([flops, params], "%.3f")
    # print(f"\n\npretrain: FLOPS: {flops}, Params: {params}\n\n")

    print('Done!')


if "__main__" == __name__:
    main()
