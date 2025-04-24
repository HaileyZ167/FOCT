import logging
import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import models
import models.backbones
import models.common
import models.sampler
from models.CT_method import CT_metrics_system, CT_AUG_system
import matplotlib.pyplot as plt
from scipy.ndimage import zoom, gaussian_filter
from torchvision import transforms
import random

LOGGER = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class PatchCore(torch.nn.Module):
    def __init__(self, device):
        """PatchCore anomaly detection class."""
        super(PatchCore, self).__init__()
        self.device = device
        self.transform_img = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def load(
        self,
        backbone,
        layers_to_extract_from,
        device,
        input_shape,
        pretrain_embed_dimension,
        target_embed_dimension,
        patchsize=3,
        patchstride=1,
        anomaly_score_num_nn=1,
        featuresampler=models.sampler.IdentitySampler(),
        nn_method=models.common.FaissNN(False, 4),
        k_fg=10,
        k_bg=40,
        query_p=0.25,
        bg_p=0.01,    
        length=100,
        rho=0.5,
        feature_path="",
        **kwargs,
    ):
        
        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape

        self.device = device
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)

        self.k_fg = k_fg
        self.k_bg = k_bg

        self.feature_path = feature_path
        self.length = length
        self.forward_modules = torch.nn.ModuleDict({})
        
        feature_aggregator = models.common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator
        
        preprocessing = models.common.Preprocessing(
            feature_dimensions, pretrain_embed_dimension
        )
        self.forward_modules["preprocessing"] = preprocessing

        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = models.common.Aggregator(
            target_dim=target_embed_dimension
        )

        _ = preadapt_aggregator.to(self.device)

        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator
       
        self.anomaly_scorer = models.common.NearestNeighbourScorer(
            n_nearest_neighbours=anomaly_score_num_nn, nn_method=nn_method
        )

        self.anomaly_segmentor = models.common.RescaleSegmentor(
            device=self.device, target_size=input_shape[-2:]
        )

        self.featuresampler = featuresampler

        # ======= CT System ==========
        self.CT_metric = CT_metrics_system(rho=rho)
        self.CT_aug = CT_AUG_system(rho=rho)

        # ======= Query sampler ==========
        self.query_p = query_p
        self.query_sampler = models.sampler.ApproximateGreedyCoresetSampler(self.query_p, device)

        self.bg_p = bg_p
        self.bg_sampler = models.sampler.ApproximateGreedyCoresetSampler(self.bg_p, device)

    def embed(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                with torch.no_grad():
                    input_image = image.to(torch.float).to(self.device)
                    features.append(self._embed(input_image))
            return features
        return self._embed(data)

    def _embed(self, images, detach=True, provide_patch_shapes=False):
        """Returns feature embeddings for images."""

        def _detach(features):
            if detach:
                return [x.detach().cpu().numpy() for x in features]
            return features
        
        _ = self.forward_modules["feature_aggregator"].eval()
        with torch.no_grad():
            features = self.forward_modules["feature_aggregator"](images)
        
        # [B, D_1, P_1, P_1]  [B, D_2, P_2, P_2] (2,512,28,28) (2,1024,14,14)
        features = [features[layer] for layer in self.layers_to_extract_from]

        if self.backbone.name == "vit_base":
            for i in range(len(features)):
                ##vit_base
                # (2,197,768)
                features[i] = features[i][:,1:,:]
                # (2,196,768)
                features[i] = features[i].permute(0,2,1)
                # (2,768,196)
                features[i]=features[i].reshape(-1, features[i].shape[1], 14, 14)
                # (2,768,14,14)

        # (2,784,512,3,3) (2,196,1024,3,3)
        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]
        
        # x_1: [28,28]  [14,14]
        patch_shapes = [x[1] for x in features]

        # x_0: ([784, 512, 3, 3]) ([196, 1024, 3, 3]) 
        # x :([2, 784, 512, 3, 3]) ([2, 196, 1024, 3, 3])
        features = [x[0] for x in features]
        
        # [28,28]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            # [2, 196, 1024, 3, 3]
            _features = features[i]
            # [14,14]
            patch_dims = patch_shapes[i]

        #     # TODO(pgehler): Add comments
            # [2, 14, 14, 1024, 3, 3]
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            # [2,  1024, 3, 3, 14, 14]
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape

            # [18432, 14, 14]
            _features = _features.reshape(-1, *_features.shape[-2:])

            # [18432, 1, 28, 28]
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            # [18432, 28, 28]
            _features = _features.squeeze(1)
            # [2,  1024, 3, 3, 28, 28]
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            # [2, 28, 28, 1024, 3, 3]
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            # [2, 784, 1024, 3, 3]
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features

        # [1568, 512, 3, 3]  [1568, 1024, 3, 3]
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]
        
        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        # [2,1024]
        features = self.forward_modules["preprocessing"](features)
        # [1024]
        features = self.forward_modules["preadapt_aggregator"](features)

        if provide_patch_shapes:
            return _detach(features), patch_shapes
        return _detach(features)

    def fit(self, training_data, testing_data, matrix_save_path):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        return self._fill_memory_bank(training_data, testing_data, matrix_save_path)

    def _fill_memory_bank(self, input_data, testing_data, matrix_save_path):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(self.device)
                return self._embed(input_image)

        def _image_to_mask(fg_mask):
            fg_mask = fg_mask.to(self.device)
            B, C, H, W = fg_mask.shape
            downsample_mask = F.interpolate(fg_mask, size=(H//8, W//8), mode='nearest')
            downsample_mask = downsample_mask.squeeze()
            downsample_mask[downsample_mask != 0] = 1
            downsample_mask = downsample_mask.view(-1)

            indices_ones = torch.nonzero(downsample_mask == 1).squeeze()
            indices_zeros = torch.nonzero(downsample_mask == 0).squeeze()

            return downsample_mask, indices_ones, indices_zeros

        s_features_list = []
        fg_features_list = []
        bg_features_list = []
        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for i, support_image in enumerate(data_iterator):
                if isinstance(support_image, dict):
                    org_image = support_image["image"]
                    fg_mask = support_image["fg_mask"]

                org_feature = _image_to_features(org_image)
                org_tensor = torch.tensor(org_feature).to(self.device)

                fg_mask, indices_ones, indices_zeros = _image_to_mask(fg_mask)
                fg_feature = torch.index_select(org_tensor, dim=0, index=indices_ones)
                bg_feature = torch.index_select(org_tensor, dim=0, index=indices_zeros)

                s_features_list.append(org_feature)
                fg_features_list.append(fg_feature)
                bg_features_list.append(bg_feature)

        fg_features = torch.cat((fg_features_list))
        bg_features = torch.cat((bg_features_list))

        K, P, D = len(s_features_list), len(org_feature), len(org_feature[1])
        s_features = np.array(s_features_list).reshape(K*P, D)
        s_features = torch.from_numpy(s_features)

        features = s_features.numpy()
        np.save(os.path.join(matrix_save_path ,"total_features.npy"), features)

        fg_features = self.featuresampler.run(fg_features)
        bg_features = self.bg_sampler.run(bg_features)

        self.fg_shape = fg_features.shape[0]
        self.bg_shape = bg_features.shape[0]
        torch.save(fg_features, os.path.join(matrix_save_path ,"coreset_fg.pt"))
        torch.save(bg_features, os.path.join(matrix_save_path, "coreset_bg.pt"))

    def select_features(self, matrix, features, k):
        """Select top-k features by transport matrix."""
        max_values, max_indices = torch.max(matrix, dim=1)
        max_dict = {max_indices[i].item(): max_values[i].item() for i in range(matrix.size(0))}
        sorted_dict = dict(sorted(max_dict.items(), key=lambda x: x[1], reverse=True))
        topk_indices = [index for index in list(sorted_dict.keys())[:k]]
        topk_indices = torch.tensor(topk_indices).to(self.device)
        selected_vectors = torch.index_select(features, dim=0, index=topk_indices)
        return selected_vectors, topk_indices

    def predict(self, data, matrix_save_path):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data, matrix_save_path)
        return self._predict(data, matrix_save_path)

    def _predict_dataloader(self, dataloader, matrix_save_path):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()
        fg_features = torch.load(os.path.join(matrix_save_path ,"coreset_fg.pt")).to(self.device)
        bg_features = torch.load(os.path.join(matrix_save_path ,"coreset_bg.pt")).to(self.device)
        total_features = torch.cat((fg_features, bg_features), dim=0)
        Q_fg = []
        Q_bg = []
        self.M_org = total_features
        self.M_fg = fg_features
        self.M_aug = total_features
        self.Q_fg = Q_fg
        self.Q_bg = Q_bg
        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        img_path_list = []
        fgmaps=[]
        bgmaps=[]
        b_maps=[]

        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for i,image in enumerate(data_iterator):
                img_path_list.append(image['image_path'])
                if isinstance(image, dict):
                    labels_gt.extend(image["is_anomaly"].numpy().tolist())
                    masks_gt.extend(image["mask"].numpy().tolist())
                    image = image["image"]
                _scores, _masks, _fgvis, _bgvis, _bmapvis = self.fs_predict(image, fg_features, bg_features, total_features)
                for score, mask, fgmap, bgmap, b_map in zip(_scores, _masks, _fgvis, _bgvis, _bmapvis):
                    scores.append(score)
                    masks.append(mask)
                    fgmaps.append(fgmap)
                    bgmaps.append(bgmap)
                    b_maps.append(b_map)

        return scores, masks, labels_gt, masks_gt, img_path_list, self.M_all, fgmaps, bgmaps, b_maps

    def _predict(self, images, matrix_save_path):
        """Infer score and mask for a batch of images."""
        images = images.to(torch.float).to(self.device)
        _ = self.forward_modules.eval()

        batchsize = images.shape[0]
        with torch.no_grad():
            features, patch_shapes = self._embed(images,provide_patch_shapes=True)
            features = np.asarray(features)
            patch_scores = image_scores = self.anomaly_scorer.predict([features])[0]
            image_scores = self.patch_maker.unpatch_scores(
                image_scores, batchsize=batchsize
            )
            # (2,784,1)
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            # (2,)
            image_scores = self.patch_maker.score(image_scores)
            # print("image_scores:",image_scores)
            # (2,784)
            patch_scores = self.patch_maker.unpatch_scores(
                patch_scores, batchsize=batchsize
            )
            scales = patch_shapes[0]
            # (2,28,28)
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])
            # (2, 224, 224)
            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)
        return [score for score in image_scores], [mask for mask in masks]

    def fs_predict(self, images, fg_features, bg_features, total_features):
        """Infer score and mask under few-shot setting."""
        images = images.to(torch.float).to(self.device)
        _ = self.forward_modules.eval()

        batchsize = images.shape[0]
       
        query_features, patch_shapes = self._embed(images, provide_patch_shapes=True)
        query_features = np.asarray(query_features)
        q_0 = torch.tensor(query_features).to(self.device)
        fg_vis = []
        bg_vis = []
        bmap_vis = []
        score_org, b_map = self.CT_aug(total_features, q_0)
        b_map_vis = b_map.sum(0)
        b_map_vis = b_map_vis.view(1, 42, 42).unsqueeze(0)
        b_map_vis = F.interpolate(b_map_vis, size=(336, 336), mode='bilinear', align_corners=False)
        b_map_vis = b_map_vis.squeeze().cpu()
        b_map_vis = np.asarray(b_map_vis)
        smoothed_b_map = gaussian_filter(b_map_vis, sigma=4)
        bmap_vis.append(smoothed_b_map)

        if self.k_fg > 0 or self.k_bg > 0:
            b_map_fg = b_map[:self.fg_shape, :]
            b_map_bg = b_map[self.fg_shape:, :]
            q_1_fg, fg_indices = self.select_features(b_map_fg, q_0, k=self.k_fg)
            q_1_bg, bg_indices = self.select_features(b_map_bg, q_0, k=self.k_bg)

            # ====== Feature AUG ======
            self.Q_fg.append(q_1_fg)
            self.Q_bg.append(q_1_bg)
            Q_aug_fg = self.Q_fg
            Q_aug_bg = self.Q_bg
            Q_c_fg = torch.cat((Q_aug_fg)).to(self.device)
            Q_c_bg = torch.cat((Q_aug_bg)).to(self.device)

            # ====== Feature Compress ======
            while Q_c_fg.shape[0] + Q_c_bg.shape[0] > self.length:
                Q_c_fg = self.query_sampler.run(Q_c_fg)
                Q_c_bg = self.query_sampler.run(Q_c_bg)

            M_fg = torch.cat((fg_features, Q_c_fg), dim=0)
            M_bg = torch.cat((bg_features, Q_c_bg), dim=0)
            N = M_fg.shape[0]
            self.M_all = torch.cat((M_fg, M_bg), dim=0)

        score, score_fg = self.CT_metric(self.M_all, q_0, N)
        # ====== anomaly map ======
        score_patches_temp = score.detach().cpu()
        anomaly_scores = np.mat(score_patches_temp)
        patch_scores = image_scores = anomaly_scores

        # ====== Image-level AUROC ====== 
        image_scores = self.patch_maker.unpatch_scores(
            image_scores, batchsize=batchsize
        )
        image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
        image_scores = self.patch_maker.score(image_scores)
        
        # ====== Pixel-level AUROC ======
        scales = patch_shapes[0]
        patch_scores = np.reshape(patch_scores,(batchsize, scales[0], scales[1]))
        patch_scores = torch.from_numpy(patch_scores)
        patch_scores = patch_scores.unsqueeze(0)
        anomaly_map = self.anomaly_segmentor.convert_to_segmentation(patch_scores)
        
        return [score for score in image_scores], [mask for mask in anomaly_map], fg_vis, bg_vis, b_map_vis

    @staticmethod
    def _params_file(filepath, prepend=""):
        return os.path.join(filepath, prepend + "patchcore_params.pkl")

    def save_to_path(self, save_path: str, prepend: str = "") -> None:
        LOGGER.info("Saving PatchCore data.")
        self.anomaly_scorer.save(
            save_path, save_features_separately=False, prepend=prepend
        )
        patchcore_params = {
            "backbone.name": self.backbone.name,
            "layers_to_extract_from": self.layers_to_extract_from,
            "input_shape": self.input_shape,
            "pretrain_embed_dimension": self.forward_modules[
                "preprocessing"
            ].output_dim,
            "target_embed_dimension": self.forward_modules[
                "preadapt_aggregator"
            ].target_dim,
            "patchsize": self.patch_maker.patchsize,
            "patchstride": self.patch_maker.stride,
            "anomaly_scorer_num_nn": self.anomaly_scorer.n_nearest_neighbours,
        }
        with open(self._params_file(save_path, prepend), "wb") as save_file:
            pickle.dump(patchcore_params, save_file, pickle.HIGHEST_PROTOCOL)

    def load_from_path(
        self,
        load_path: str,
        device: torch.device,
        nn_method: models.common.FaissNN(False, 4),
        prepend: str = "",
    ) -> None:
        LOGGER.info("Loading and initializing PatchCore.")
        with open(self._params_file(load_path, prepend), "rb") as load_file:
            patchcore_params = pickle.load(load_file)
        patchcore_params["backbone"] = models.backbones.load(
            patchcore_params["backbone.name"]
        )
        patchcore_params["backbone"].name = patchcore_params["backbone.name"]
        del patchcore_params["backbone.name"]
        self.load(**patchcore_params, device=device, nn_method=nn_method)

        self.anomaly_scorer.load(load_path, prepend)


# Image handling classes.
class PatchMaker:
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 1:
            x = torch.max(x, dim=-1).values
        if was_numpy:
            return x.numpy()
        return x
