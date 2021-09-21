import os
import json
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.transforms.functional import scale

from PIK3CA_mutation.data_reader import ClsDataset
from PIK3CA_mutation.utils import get_modelpath, get_scalerpath, get_thresholdpath, net_prediction_oneshop, patient_res_m3_oneshop, save_results

from sklearn.preprocessing import StandardScaler
from glob import glob
from scipy import io
from torchvision import models, transforms
import joblib


model_name_list = ['PIK3CA_mutation_0',
                   'PIK3CA_mutation_1', 'PIK3CA_mutation_2']
patient_scaler_name_list = ['patient_scaler_0',
                            'patient_scaler_1', 'patient_scaler_2']


def standard_scale(data, testingflag, scaler_path):
    if testingflag:
        scaler = joblib.load(scaler_path)
    else:
        scaler = StandardScaler()
        scaler.fit(data)

        print(scaler.mean_)
        print(scaler.var_)
        joblib.dump(scaler, os.path.join(scaler_path))

    return scaler.transform(data)


def start_model(datapath, sampling_file, root_dir, model, seed=2020, gpu="0", net="resnet18",
                num_classes=2, num_workers=4, batch_size=256, norm_mean=[0.8201, 0.5207, 0.7189],
                norm_std=[0.1526, 0.1542, 0.1183]):
    """
    Arguments:
      model: PIK3CA_Mutation_0, 'PIK3CA_Mutation_1, PIK3CA_Mutation_2
      net: resnet18, alexnet, resnet34, inception_v3

    Results:
      root_dir: ./FUSCC001_models/
      patch.json: ${root_dir}/${model}/patch.json
      patch.npz: ${root_dir}/${model}/patch.npz
      patient.json: ${root_dir}/${model}/patient.json
      patient.npz: ${root_dir}/${model}/patient.npz
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        # Operated on original image, rewrite on previous transform.
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)])

    print('Loading data...')
    testset = ClsDataset(sampling_file, datapath, preprocess)
    testloader = DataLoader(testset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)

    net = getattr(models, net)(pretrained=False, num_classes=num_classes)
    # 这个函数不是看的很懂，但是按照现在的代码后面的pkl应该是不需要了？
    modelpath = get_modelpath(model)
    print('Loading model...', modelpath)

    if len(gpu) > 1:
        net = torch.nn.DataParallel(net).cuda()
        # load the finetune weight parameters
        net.load_state_dict(torch.load(modelpath))
    else:
        net = net.cuda()
        net.load_state_dict(
            {k.replace('module.', ''): v for k, v in torch.load(modelpath).items()})

    # Patch Output: patch.json / patch.npz
    scores_patch, predictions_patch, namelist_patch = net_prediction_oneshop(
        testloader, net, num_classes)

    patch_results = save_results(
        namelist_patch, scores_patch[:, 1], predictions_patch, num_classes)
    with open(os.path.join(root_dir, model, 'patch.json'), 'w') as f:
        json.dump(patch_results, f)

    savename_patch = os.path.join(root_dir, model, 'patch.npz')
    np.savez(savename_patch, key_score=scores_patch,
             key_binpred=predictions_patch, key_namelist=namelist_patch)

    # Patient Output: patient.json / patient.npz
    scores_patient, predictions_patient, namelist_patient = patient_res_m3_oneshop(scores_patch, namelist_patch,
                                                                                   num_classes)
    patient_results = save_results(
        namelist_patient, scores_patient[:, 1], predictions_patient, num_classes)
    with open(os.path.join(root_dir, model, 'patient.json'), 'w') as f:
        json.dump(patient_results[0], f)

    savename_patient = os.path.join(root_dir, model, 'patient.npz')
    np.savez(savename_patient, key_score=scores_patient,
             key_binpred=predictions_patient, key_namelist=namelist_patient)

    with open(os.path.join(root_dir, model, 'prediction.json'), 'w') as f:
        results = {
            "model": model,
            "patient": patient_results[0],
            "patch": patch_results
        }
        json.dump(results, f)


def fuse_res(root_dir):
    threshold_file = get_thresholdpath('threshold_patient')

    for i in range(len(model_name_list)):

        patient_npz_file = os.path.join(
            root_dir, model_name_list[0], 'patient.npz')
        npz_res = np.load(patient_npz_file)

        scores = npz_res['key_score'][:, 1].reshape(-1, 1)
        names = npz_res['key_namelist']
        scaler = get_scalerpath(patient_scaler_name_list[i])

        scaled_score = standard_scale(data=scores,
                                      testingflag=True,
                                      scaler_path=scaler)

        if i == 0:
            all_scores = scaled_score

        else:
            all_scores = np.concatenate((all_scores, scaled_score), axis=1)

    mean_scores = np.mean(all_scores, axis=1)

    threshold_patient_crossval = float(np.load(threshold_file)['threshold_patient_crossval'])

    bins = np.array(
        [1 if score >= threshold_patient_crossval else 0 for score in mean_scores])

    return mean_scores, bins


def start_models(datapath, sampling_file, root_dir, seed=2020, gpu="0", net="resnet18",
                 num_classes=2, num_workers=4, batch_size=256, norm_mean=[0.8201, 0.5207, 0.7189],
                 norm_std=[0.1526, 0.1542, 0.1183]):
    """
    Arguments:
      model_root: a folder containing all models: model_root\\task_name\\model files
      res_root: a folder containing all predictin results:  res_root\\task_name\\model_name\\results files
      net: resnet18, alexnet, resnet34, inception_v3
      root_dir: path to save the results

    Results: ##跑完循环之后，总共会产生3个以模型名称命名的文件夹，每个文件夹下都会有一个patch.npz和一个patient.npz
      patch.npz: ${root_dir}/${model1}/patch.npz
      patch.npz: ${root_dir}/${model2}/patch.npz
      patch.npz: ${root_dir}/${model3}/patch.npz

      patient.npz: ${root_dir}/${model1}/patient.npz
      patient.npz: ${root_dir}/${model2}/patient.npz
      patient.npz: ${root_dir}/${model3}/patient.npz

    """
    for model in ['PIK3CA_mutation_0', 'PIK3CA_mutation_1', 'PIK3CA_mutation_2']:
        start_model(datapath, sampling_file, root_dir, model, seed=seed, gpu=gpu, net=net,
                    num_classes=num_classes, num_workers=num_workers, batch_size=batch_size,
                    norm_mean=norm_mean, norm_std=norm_std)

    fuse_res(root_dir)
