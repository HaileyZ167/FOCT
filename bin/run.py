import contextlib
import logging
import os
import sys
import time

cur_path=os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, cur_path+"/..")

import click
import numpy as np
import torch

import models.backbones
import models.common
import models.metrics
import models.memory_update
import models.sampler
import models.utils
import cv2
import json

LOGGER = logging.getLogger(__name__)
_DATASETS = {"mvtec": ["models.datasets.fs_mvtec", "MVTec_Dataset"]}
_DATASETS = {"mpdd": ["models.datasets.fs_mpdd", "MPDD_Dataset"]}
@click.group(chain=True)
@click.argument("results_path", type=str)
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--log_group", type=str, default="group")
@click.option("--log_project", type=str, default="project")
@click.option("--save_segmentation_images", default=True, is_flag=True)
@click.option("--save_patchcore_model", is_flag=True)
def main(**kwargs):
    pass

def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)

def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=np.float32)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)

def vis_output(img_path_list, preds, labels, vis_dir):
    for i in range(len(img_path_list)):
        img_path = img_path_list[i][0]
        filedir, filename = os.path.split(img_path)
        filedir, defename = os.path.split(filedir)
        filedir, _ = os.path.split(filedir)
        _, clsname = os.path.split(filedir)
        save_dir = os.path.join(vis_dir, clsname, defename)
        # print(save_dir)
        os.makedirs(save_dir, exist_ok=True)

        # image
        image = cv2.imread(img_path)
        image = cv2.resize(image, (368, 368))
        x = (368 - 336) // 2
        y = (368 - 336) // 2
        image = image[y:y + 336, x:x + 336]

        mask = labels[i]
        mask = (mask * 255).astype(np.uint8).transpose((1,2,0)).repeat(3, 2)

        # anomaly map
        anomaly_map = preds[i]
        anomaly_map = anomaly_map[:,:,None].repeat(3, 2)
        # print(mask.shape, anomaly_map.shape)

        scoremap_self = apply_ad_scoremap(image, normalize(anomaly_map))
        scoremap_self = cv2.cvtColor(scoremap_self, cv2.COLOR_RGB2BGR)

        # save
        save_path = os.path.join(save_dir, filename)
        save_path1 = os.path.join(save_dir, "all")
        os.makedirs(save_path1, exist_ok=True)
        save_path1 = os.path.join(save_path1, filename)
        scoremap = np.hstack([image, mask, scoremap_self])
        cv2.imwrite(save_path, scoremap_self)
        cv2.imwrite(save_path1, scoremap)

def vis_output1(img_path_list, preds1, preds2, labels, vis_dir):
    for i in range(len(img_path_list)):
        img_path = img_path_list[i][0]
        filedir, filename = os.path.split(img_path)
        filedir, defename = os.path.split(filedir)
        filedir, _ = os.path.split(filedir)
        _, clsname = os.path.split(filedir)
        save_dir = os.path.join(vis_dir, clsname, defename)
        # print(save_dir)
        os.makedirs(save_dir, exist_ok=True)

        # image
        image = cv2.imread(img_path)
        image = cv2.resize(image, (368, 368))
        x = (368 - 336) // 2
        y = (368 - 336) // 2
        image = image[y:y + 336, x:x + 336]

        mask = labels[i]
        mask = (mask * 255).astype(np.uint8).transpose((1,2,0)).repeat(3, 2)

        # anomaly map
        anomaly_map1 = preds1[i]
        anomaly_map1 = anomaly_map1[:,:,None].repeat(3, 2)
        # print(mask.shape, anomaly_map.shape)

        scoremap_self1 = apply_ad_scoremap(image, normalize(anomaly_map1))
        scoremap_self1 = cv2.cvtColor(scoremap_self1, cv2.COLOR_RGB2BGR)
        # bg_map
        anomaly_map2 = preds2[i]
        anomaly_map2 = anomaly_map2[:,:,None].repeat(3, 2)

        scoremap_self2 = apply_ad_scoremap(image, normalize(anomaly_map2))
        scoremap_self2 = cv2.cvtColor(scoremap_self2, cv2.COLOR_RGB2BGR)
        # save
        save_path = os.path.join(save_dir, filename)
        save_path1 = os.path.join(save_dir, "all")
        os.makedirs(save_path1, exist_ok=True)
        save_path1 = os.path.join(save_path1, filename)
        scoremap = np.hstack([scoremap_self1, scoremap_self2])
        scoremap_s = np.hstack([image, mask, scoremap_self1, scoremap_self2])
        cv2.imwrite(save_path, scoremap)
        cv2.imwrite(save_path1, scoremap_s)


@main.result_callback()
def run(
    methods,
    results_path,
    gpu,
    seed,
    log_group,
    log_project,
):
    methods = {key: item for (key, item) in methods}

    run_save_path = models.utils.create_storage_folder(
        results_path, log_project, log_group, mode="iterate"
    )

    list_of_dataloaders = methods["get_dataloaders"](seed)

    device = models.utils.set_torch_device(gpu)
    # Device context here is specifically set and used later
    # because there was GPU memory-bleeding which I could only fix with
    # context managers.
    device_context = (
        torch.cuda.device("cuda:{}".format(device.index))
        if "cuda" in device.type.lower()
        else contextlib.suppress()
    )

    result_collect = []

    for dataloader_count, dataloaders in enumerate(list_of_dataloaders):
        LOGGER.info(
            "Evaluating dataset [{}] ({}/{})...".format(
                dataloaders["training"].name,
                dataloader_count + 1,
                len(list_of_dataloaders),
            )
        )
        
        models.utils.fix_seeds(seed, device)

        dataset_name = dataloaders["training"].name

        with device_context:
            torch.cuda.empty_cache()
            imagesize = dataloaders["training"].dataset.imagesize
            classname = dataloaders["training"].dataset.classname
            matrix_save_path = os.path.join(run_save_path, "models", "mpdd_" + classname)
            os.makedirs(matrix_save_path, exist_ok=True)
            sampler = methods["get_sampler"](
                device,
            )
            PatchCore_list = methods["get_patchcore"](imagesize, sampler, device)   # patchcore_update

            if len(PatchCore_list) > 1:
                LOGGER.info(
                    "Utilizing PatchCore Ensemble (N={}).".format(len(PatchCore_list))
                )

            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                if PatchCore.backbone.seed is not None:
                    models.utils.fix_seeds(PatchCore.backbone.seed, device)
                LOGGER.info(
                    "Training models ({}/{})".format(i + 1, len(PatchCore_list))
                )
                torch.cuda.empty_cache()
                PatchCore.fit(
                    dataloaders["training"], dataloaders["testing"], matrix_save_path
                    )

            
            torch.cuda.empty_cache()
            aggregator = {"scores": [], "segmentations": []}
            feature_maps = {"fg_maps": [], "bg_maps": [], "b_maps": []}
            start_time = time.time()
            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                LOGGER.info(
                    "Embedding test data with models ({}/{})".format(
                        i + 1, len(PatchCore_list)
                    )
                )
                # list scores /segmentations
                scores, segmentations, labels_gt, masks_gt, img_path_list, memory, fg_map_list, bg_map_list, bmap_list = PatchCore.predict(
                    dataloaders["testing"], matrix_save_path
                )
                print("memeory:", memory.shape)
                feature_maps["fg_maps"].append(fg_map_list)
                feature_maps["bg_maps"].append(bg_map_list)
                feature_maps["b_maps"].append(bmap_list)

                aggregator["scores"].append(scores)
                aggregator["segmentations"].append(segmentations)
            print('Used time:',time.time() - start_time)

            scores = np.array(aggregator["scores"])
            min_scores = scores.min(axis=-1).reshape(-1, 1)
            max_scores = scores.max(axis=-1).reshape(-1, 1)
            scores = (scores - min_scores) / (max_scores - min_scores)
            scores = np.mean(scores, axis=0)

            segmentations = np.array(aggregator["segmentations"])
            min_scores = (
                segmentations.reshape(len(segmentations), -1)
                .min(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            max_scores = (
                segmentations.reshape(len(segmentations), -1)
                .max(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            segmentations = (segmentations - min_scores) / (max_scores - min_scores)
            segmentations = np.mean(segmentations, axis=0)

            # fg_maps
            fg_maps = np.array(feature_maps["fg_maps"])
            min_scores = (
                fg_maps.reshape(len(fg_maps), -1)
                .min(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            max_scores = (
                fg_maps.reshape(len(fg_maps), -1)
                .max(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            fg_maps = (fg_maps - min_scores) / (max_scores - min_scores)
            fg_maps = np.mean(fg_maps, axis=0)
            # bg_maps
            fg_maps = np.array(feature_maps["fg_maps"])
            min_scores = (
                fg_maps.reshape(len(fg_maps), -1)
                .min(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            max_scores = (
                fg_maps.reshape(len(fg_maps), -1)
                .max(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            fg_maps = (fg_maps - min_scores) / (max_scores - min_scores)
            fg_maps = np.mean(fg_maps, axis=0)

            # bg_maps
            bg_maps = np.array(feature_maps["bg_maps"])
            min_scores = (
                bg_maps.reshape(len(bg_maps), -1)
                .min(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            max_scores = (
                bg_maps.reshape(len(bg_maps), -1)
                .max(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            bg_maps = (bg_maps - min_scores) / (max_scores - min_scores)
            bg_maps = np.mean(bg_maps, axis=0)

            # b_maps
            b_maps = np.array(feature_maps["b_maps"])
            min_scores = (
                b_maps.reshape(len(b_maps), -1)
                .min(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            max_scores = (
                b_maps.reshape(len(b_maps), -1)
                .max(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            b_maps = (b_maps - min_scores) / (max_scores - min_scores)
            b_maps = np.mean(b_maps, axis=0)

            anomaly_labels = [
                x[1] != "good" for x in dataloaders["testing"].dataset.data_to_iterate
            ]

            # (Optional) Plot example images.
            save_segmentation_images = False
            if save_segmentation_images:
                image_paths = [
                    x[2] for x in dataloaders["testing"].dataset.data_to_iterate
                ]
                mask_paths = [
                    x[3] for x in dataloaders["testing"].dataset.data_to_iterate
                ]

                def image_transform(image):
                    in_std = np.array(
                        dataloaders["testing"].dataset.transform_std
                    ).reshape(-1, 1, 1)
                    in_mean = np.array(
                        dataloaders["testing"].dataset.transform_mean
                    ).reshape(-1, 1, 1)
                    image = dataloaders["testing"].dataset.transform_img(image)
                    return np.clip(
                        (image.numpy() * in_std + in_mean) * 255, 0, 255
                    ).astype(np.uint8)

                def mask_transform(mask):
                    return dataloaders["testing"].dataset.transform_mask(mask).numpy()

                image_save_path = os.path.join(
                    run_save_path, "segmentation_images", dataset_name
                )
                os.makedirs(image_save_path, exist_ok=True)
                models.utils.plot_segmentation_images(
                    image_save_path,
                    classname,
                    image_paths,
                    segmentations,
                    scores,
                    mask_paths,
                    image_transform=image_transform,
                    mask_transform=mask_transform,
                )

            LOGGER.info("Computing evaluation metrics.")

            metrics = models.metrics.compute_imagewise_retrieval_metrics(
                scores, anomaly_labels
                )
            auroc = metrics["auroc"]
            image_f1_max = metrics["f1_max"]
            save_path = os.path.join(run_save_path, "visa_" + classname)
            os.makedirs(save_path, exist_ok=True)
            models.utils.plot_roc_curve(metrics['fpr'], metrics['tpr'], save_path=os.path.join(save_path, "roc.png"))

            # Compute PRO score & PW Auroc for all images
            pixel_scores = models.metrics.compute_pixelwise_retrieval_metrics(
                segmentations, masks_gt
            )
            full_pixel_auroc = pixel_scores["auroc"]
            pixel_f1_max = pixel_scores["f1_max"]
            segmentations_arr = np.array(segmentations)
            masks_gt_arr = np.array(masks_gt)
            vis_path = os.path.join(run_save_path,"vis")
            os.makedirs(vis_path, exist_ok=True)
            vis_output(img_path_list, segmentations_arr, masks_gt_arr, vis_path)

            sel_idxs = []
            for i in range(len(masks_gt)):
                if np.sum(masks_gt[i]) > 0:
                    sel_idxs.append(i)
            pixel_scores = models.metrics.compute_pixelwise_retrieval_metrics(
                [segmentations[i] for i in sel_idxs],
                [masks_gt[i] for i in sel_idxs],
            )
            anomaly_pixel_auroc = pixel_scores["auroc"]

            result_collect.append(
                {
                    "dataset_name": dataset_name,
                    "instance_auroc": auroc,
                    "full_pixel_auroc": full_pixel_auroc,
                    "anomaly_pixel_auroc": anomaly_pixel_auroc,
                    "instance_f1_max": image_f1_max,
                    "pixel_f1_max": pixel_f1_max,
                    
                }
            )
            results_record = {
                    "dataset_name": dataset_name,
                    "instance_auroc": auroc,
                    "full_pixel_auroc": full_pixel_auroc,
                    "anomaly_pixel_auroc": anomaly_pixel_auroc,
                    "instance_f1_max": image_f1_max,
                    "pixel_f1_max": pixel_f1_max,
                    
                }
            json_save_path = os.path.join(save_path, "results.json")
            with open(json_save_path, "w") as file:
                json.dump(results_record, file)

            for key, item in result_collect[-1].items():
                if key != "dataset_name":
                    LOGGER.info("{0}: {1:3.3f}".format(key, item))

            # (Optional) Store PatchCore model for later re-use.
            # SAVE all patchcores only if mean_threshold is passed?
            save_patchcore_model = False
            if save_patchcore_model:
                patchcore_save_path = os.path.join(
                    run_save_path, "models", dataset_name
                )
                os.makedirs(patchcore_save_path, exist_ok=True)
                for i, PatchCore in enumerate(PatchCore_list):
                    prepend = (
                        "Ensemble-{}-{}_".format(i + 1, len(PatchCore_list))
                        if len(PatchCore_list) > 1
                        else ""
                    )
                    PatchCore.save_to_path(patchcore_save_path, prepend)

        LOGGER.info("\n\n-----\n")

    # Store all results and mean scores to a csv-file.
    result_metric_names = list(result_collect[-1].keys())[1:]
    result_dataset_names = [results["dataset_name"] for results in result_collect]
    result_scores = [list(results.values())[1:] for results in result_collect]
    models.utils.compute_and_store_final_results(
        run_save_path,
        result_scores,
        column_names=result_metric_names,
        row_names=result_dataset_names,
    )


@main.command("patch_core")
# Pretraining-specific parameters.
@click.option("--backbone_names", "-b", type=str, multiple=True, default=[])
@click.option("--layers_to_extract_from", "-le", type=str, multiple=True, default=[])
# Parameters for Glue-code (to merge different parts of the pipeline.
@click.option("--pretrain_embed_dimension", type=int, default=1024)
@click.option("--target_embed_dimension", type=int, default=1024)
@click.option("--preprocessing", type=click.Choice(["mean", "conv"]), default="mean")
@click.option("--aggregation", type=click.Choice(["mean", "mlp"]), default="mean")
# Nearest-Neighbour Anomaly Scorer parameters.
@click.option("--anomaly_scorer_num_nn", type=int, default=5)
# Patch-parameters.
@click.option("--patchsize", type=int, default=3)
@click.option("--patchscore", type=str, default="max")
@click.option("--patchoverlap", type=float, default=0.0)
@click.option("--patchsize_aggregate", "-pa", type=int, multiple=True, default=[])
# NN on GPU.
@click.option("--faiss_on_gpu", is_flag=True)
@click.option("--faiss_num_workers", type=int, default=8)
# num of top-k query features in transport matrix
@click.option("--k_fg", type=int, default=20, show_default=True)
@click.option("--k_bg", type=int, default=40, show_default=True)
@click.option("--query_p", type=float, default=0.25, show_default=True)
@click.option("--bg_p", type=float, default=0.25, show_default=True)
# max length of memory
@click.option("--rho", type=float, default=0.1, show_default=True)
@click.option("--length", type=int, default=100, show_default=True)
# vis feature path
@click.option("--feature_path", type=str, default="", show_default=True)
def patch_core(
    backbone_names,
    layers_to_extract_from,
    pretrain_embed_dimension,
    target_embed_dimension,
    preprocessing,
    aggregation,
    patchsize,
    patchscore,
    patchoverlap,
    anomaly_scorer_num_nn,
    patchsize_aggregate,
    faiss_on_gpu,
    faiss_num_workers,
    aug_p,
    k_fg,
    k_bg,
    query_p,
    bg_p,
    rho,
    length,
    feature_path,
):
    backbone_names = list(backbone_names)
    if len(backbone_names) > 1:
        layers_to_extract_from_coll = [[] for _ in range(len(backbone_names))]
        for layer in layers_to_extract_from:
            idx = int(layer.split(".")[0])
            layer = ".".join(layer.split(".")[1:])
            layers_to_extract_from_coll[idx].append(layer)
    else:
        layers_to_extract_from_coll = [layers_to_extract_from]


    def get_patchcore(input_shape, sampler, device):
        loaded_patchcores = []
        for backbone_name, layers_to_extract_from in zip(
            backbone_names, layers_to_extract_from_coll
        ):
            backbone_seed = None
            if ".seed-" in backbone_name:
                backbone_name, backbone_seed = backbone_name.split(".seed-")[0], int(
                    backbone_name.split("-")[-1]
                )
            backbone = models.backbones.load(backbone_name)
            backbone.name, backbone.seed = backbone_name, backbone_seed

            nn_method = models.common.FaissNN(faiss_on_gpu, faiss_num_workers)

            patchcore_instance = models.patchcore_update.PatchCore(device)
            patchcore_instance.load(
                backbone=backbone,
                layers_to_extract_from=layers_to_extract_from,
                device=device,
                input_shape=input_shape,
                pretrain_embed_dimension=pretrain_embed_dimension,
                target_embed_dimension=target_embed_dimension,
                patchsize=patchsize,
                featuresampler=sampler,
                anomaly_scorer_num_nn=anomaly_scorer_num_nn,
                nn_method=nn_method,
                aug_p=aug_p,
                k_fg=k_fg,
                k_bg=k_bg,
                query_p = query_p,
                bg_p=bg_p,
                rho=rho,
                length=length,
                feature_path=feature_path,
            )
            loaded_patchcores.append(patchcore_instance)
        return loaded_patchcores

    return ("get_patchcore", get_patchcore)


@main.command("sampler")
@click.argument("name", type=str)
@click.option("--percentage", "-p", type=float, default=0.1, show_default=True)
def sampler(name, percentage):
    def get_sampler(device):
        if name == "identity":
            return models.sampler.IdentitySampler()
        elif name == "greedy_coreset":
            return models.sampler.GreedyCoresetSampler(percentage, device)
        elif name == "approx_greedy_coreset":
            return models.sampler.ApproximateGreedyCoresetSampler(percentage, device)

    return ("get_sampler", get_sampler)


@main.command("dataset")
@click.argument("name", type=str)
@click.argument("data_path", type=click.Path(exists=True, file_okay=False))
@click.option("--subdatasets", "-d", multiple=True, type=str, required=True)
@click.option("--batch_size", default=1, type=int, show_default=True)
@click.option("--shot", default=2, type=int, show_default=True)
@click.option("--num_workers", default=8, type=int, show_default=True)
@click.option("--resize", default=256, type=int, show_default=True)
@click.option("--imagesize", default=224, type=int, show_default=True)
def dataset(
    name,
    data_path,
    subdatasets,
    batch_size,
    shot,
    resize,
    imagesize,
    num_workers,
):
    dataset_info = _DATASETS[name]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

    def get_dataloaders(seed):
        dataloaders = []
        for subdataset in subdatasets:

            train_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=subdataset,
                split=dataset_library.DatasetSplit.TRAIN,
                resize=resize,
                imagesize=imagesize,
                shot=shot,
                seed=seed,
                batch=batch_size,
            )
            test_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=subdataset,
                resize=resize,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TEST,
                seed=seed,
                shot=shot,
            )
            print(len(test_dataset))

            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            train_dataloader.name = name
            if subdataset is not None:
                train_dataloader.name += "_" + subdataset

            dataloader_dict = {
                "training": train_dataloader,
                "testing": test_dataloader,
            }

            dataloaders.append(dataloader_dict)
        return dataloaders

    return ("get_dataloaders", get_dataloaders)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
    main()
