import demo_base
import torch

demo_base.TASK = 'msegment'
demo_base.EPOCHS = 100
demo_base.IMGSZ = 960
demo_base.BATCH_SIZE = 64
demo_base.DEVICE = torch.device('cuda:0')
demo_base.DATA = "fusedata2419_mseg_c5_0730.yaml"
# demo_base.CONF_VAL = 0.5
# demo_base.CONF_PREDICT = 0.5

MODEL1 = 'yolov8x-mseg-7.yaml'
MODEL2 = 'yolov8x-mseg-dlka3res-7.yaml'
MODEL3 = 'yolov10x-mseg-dlka3res-7.yaml'
MODEL4 = 'yolov10x-mseg-dlka3res-7-unet.yaml'
MODEL4_S = 'yolov10x-mseg-dlka3res-7-unet-single.yaml'
MODEL5 = 'yolov10x-mseg-dlka3res-7-c3str.yaml'
MODEL6 = 'yolov10x-mseg-dlka3res-7-unet-sep.yaml'
MODEL7 = 'yolov10x-mseg-dlka3res-7-c3str-unet1.yaml'
MODEL8 = 'yolov10x-mseg-dlka3res-7-c3str-unet2.yaml'
MODEL9 = 'yolov10x-mseg-dlka3res-7-dfl.yaml'
MODEL07 = 'yolov10x-mseg-dlka3res-7-dlka.yaml'
MODEL08 = 'yolov10x-mseg-dlka3res-7-dlkaatt.yaml'
SEG_WEIGHT = "runs/segment/fusedata5894_seg_c5_0822_80p-[yolov8x-seg-dlka3res]2/weights/best.pt"
SEG_WEIGHT10 = "/localnvme/project/ultralytics/runs/segment/fusedata7961_seg_c5_l2_1022_re_80p_ref-[yolov10x-seg-dlka3res]8/weights/best.pt"
SEG_WEIGHT10_v12 = r'runs/segment/fusedata7961_seg_c5_1106_v12_src-[yolov10x-seg-dlka3res]/weights/best.pt'
DATA0 = "fusedata7961_mseg_c5_l2_1104_v5_test.yaml"
DATA1 = "fusedata7961_mseg_c5_l2_1106_v10_test.yaml"

DATA_SA = "fusedata7961_mseg_c5_l2_1112_v16_sa_test.yaml"
DATA_SB = "fusedata7961_mseg_c5_l2_1112_v16_sb_test.yaml"
DATA_SC = "fusedata7961_mseg_c5_l2_1112_v16_sc_test.yaml"
DATA_SD = "fusedata7961_mseg_c5_l2_1112_v16_sd_test.yaml"

DATA_Test = 'defect_test_1021.yaml'

if __name__ == '__main__':
    pass
    # demo_base.yolo10(
    #     MODEL4_S, weight_path=SEG_WEIGHT10, data=DATA_Test, auto_optim=False, retrain=True,
    #     mloss_mask=False, mloss_weight=0, mloss_enlarge=20, name='debug'
    # )
    # demo_base.yolo10(
    #     MODEL4_S, weight_path=SEG_WEIGHT10, data=DATA_SB, auto_optim=False, retrain=True,
    #     mloss_mask=False, mloss_weight=0, mloss_enlarge=20,
    # )
    # demo_base.yolo10(
    #     MODEL4_S, weight_path=SEG_WEIGHT10, data=DATA_SD, auto_optim=False, retrain=True,
    #     mloss_mask=False, mloss_weight=0, mloss_enlarge=20,
    # )
    # demo_base.yolo10(
    #     MODEL4_S, weight_path=SEG_WEIGHT10, data=DATA_SA, auto_optim=False, retrain=True,
    #     mloss_mask=False, mloss_weight=0, mloss_enlarge=20,
    # )
    # demo_base.yolo10(
    #     MODEL4_S, weight_path=SEG_WEIGHT10, data=DATA_SC, auto_optim=False, retrain=True,
    #     mloss_mask=False, mloss_weight=0, mloss_enlarge=20,
    # )

    # demo_base.yolo10(
    #     MODEL4_S, weight_path=SEG_WEIGHT10, data=DATA_SB, auto_optim=False, retrain=True,
    #     mloss_mask=False, mloss_weight=0, mloss_enlarge=100,
    # )
    # demo_base.yolo10(
    #     MODEL4_S, weight_path=SEG_WEIGHT10, data=DATA_SB, auto_optim=False, retrain=True,
    #     mloss_mask=False, mloss_weight=0, mloss_enlarge=50,
    # )
    # demo_base.yolo10(
    #     MODEL4_S, weight_path=SEG_WEIGHT10, data=DATA_SB, auto_optim=False, retrain=True,
    #     mloss_mask=False, mloss_weight=0, mloss_enlarge=20,
    # )
    # demo_base.yolo10(
    #     MODEL4_S, weight_path=SEG_WEIGHT10, data=DATA_SB, auto_optim=False, retrain=True,
    #     mloss_mask=False, mloss_weight=0, mloss_enlarge=10,
    # )


    # val_name = 'fusedata7961_mseg_c5_l2_1029_abandonment_refine_80p_ref_src-[yolov10x-mseg-dlka3res-7-unet]2'
    # demo_base.model_val(val_name, save_txt=True, save_conf=True)
    # conf=0.5
    # demo_base.model_val(val_name, filter_small=0.05, conf=conf)
    # demo_base.model_val(val_name, filter_small=0.05, conf=conf, risk_enlarge=1.1)
    # demo_base.model_val(val_name, filter_small=0.05, conf=conf, risk_enlarge=1.2)
    # demo_base.model_val(val_name, filter_small=0.05, conf=conf, risk_enlarge=1.3)
    # demo_base.model_val(val_name, filter_small=0.05, conf=conf, risk_enlarge=1.4)
    # demo_base.model_val(val_name, filter_small=0.05, conf=conf, risk_enlarge=1.5)
    # demo_base.model_val(val_name, filter_small=0.05, conf=conf, risk_enlarge=1.6)
    # demo_base.model_val(val_name, filter_small=0.05, conf=conf, risk_enlarge=1.7)
    # demo_base.model_val(val_name, filter_small=0.05, conf=conf, risk_enlarge=1.8)
    # demo_base.model_val(val_name, filter_small=0.05, conf=conf, risk_enlarge=1.9)
    # demo_base.model_val(val_name, filter_small=0.05, conf=conf, risk_enlarge=2.0)
    # demo_base.model_val(val_name, filter_small=0.05, conf=conf, risk_enlarge=[1.0, 1.0, 2.0, 2.0], save_conf=True, save_txt=True)
    #
    # val_name = 'fusedata7961_mseg_c5_l2_1029_abandonment_refine_test-[yolov10x-mseg-dlka3res-7-unet]4'
    # val_name1 = r'fusedata7961_mseg_c5_l2_1030_v4_src-[yolov10x-mseg-dlka3res-7-unet]'
    # val_name2 = r'fusedata7961_mseg_c5_l2_1030_v4_src-[yolov10x-mseg-dlka3res-7-unet-sep]'
    # val_name3 = r'fusedata7961_mseg_c5_l2_1030_v4_test-[yolov10x-mseg-dlka3res-7-unet]2'
    # val_name4 = r'fusedata7961_mseg_c5_l2_1030_v4_test-[yolov10x-mseg-dlka3res-7-unet-sep]'

    # val_name1 = r'fusedata7961_mseg_c5_l2_1030_v4_src-[yolov10x-mseg-dlka3res-7-unet]'
    # val_name2 = r'fusedata7961_mseg_c5_l2_1030_v4_src-[yolov10x-mseg-dlka3res-7-unet-sep]'
    # val_name3 = r'fusedata7961_mseg_c5_l2_1030_v4_test-[yolov10x-mseg-dlka3res-7-unet]2'
    # val_name4 = r'fusedata7961_mseg_c5_l2_1030_v4_test-[yolov10x-mseg-dlka3res-7-unet-sep]'
    # val_name1 = r'fusedata7961_mseg_c5_l2_1031_v4_src-[yolov10x-mseg-dlka3res-7-unet]'
    # val_name2 = r'fusedata7961_mseg_c5_l2_1031_v4_src-[yolov10x-mseg-dlka3res-7-unet-sep]'
    # val_name3 = r'fusedata7961_mseg_c5_l2_1031_v4_test-[yolov10x-mseg-dlka3res-7-unet]'
    # val_name4 = r'fusedata7961_mseg_c5_l2_1031_v4_test-[yolov10x-mseg-dlka3res-7-unet-sep]'
    #
    # data1 = 'fusedata7961_mseg_c5_l2_1031_v4_src.yaml'
    # data2 = 'fusedata7961_mseg_c5_l2_1031_v4_test.yaml'
    # demo_base.model_val(val_name1, data=data1, save_txt=True, save_conf=True)
    # demo_base.model_val(val_name2, data=data1, save_txt=True, save_conf=True)
    # demo_base.model_val(val_name3, data=data1, save_txt=True, save_conf=True)
    # demo_base.model_val(val_name4, data=data1, save_txt=True, save_conf=True)
    # demo_base.model_val(val_name1, data=data2, save_txt=True, save_conf=True)
    # demo_base.model_val(val_name2, data=data2, save_txt=True, save_conf=True)
    # demo_base.model_val(val_name3, data=data2, save_txt=True, save_conf=True)
    # demo_base.model_val(val_name4, data=data2, save_txt=True, save_conf=True)

    # val_name = 'fusedata7961_mseg_c5_l2_1030_abandonment_refine_test-[yolov10x-mseg-dlka3res-7-unet]'
    # demo_base.model_val(val_name, save_txt=True, save_conf=True)
    # conf=0.3
    # demo_base.model_val(val_name, filter_small=0.05, conf=conf)                                                                    # data = 'defect_test_1023.yaml'
    # demo_base.model_val(val_name, filter_small=0.05, conf=conf, risk_enlarge=1.1)                                                  # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, risk_enlarge=[1.0, 1.3, 2.0, 1.0], data=data, save_conf=True, save_txt=True)
    # demo_base.model_val(val_name, filter_small=0.05, conf=conf, risk_enlarge=1.2)                                                  # val_name = 'fusedata7961_mseg_c5_l2_1023_src_80p_ref-[yolov10x-mseg-dlka3res-7]3'
    # demo_base.model_val(val_name, filter_small=0.05, conf=conf, risk_enlarge=1.3)                                                  # data = 'fusedata7961_mseg_c5_l2_1023_src_80p_ref.yaml'
    # demo_base.model_val(val_name, filter_small=0.05, conf=conf, risk_enlarge=1.4)                                                  # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, risk_enlarge=[1.0, 1.3, 2.0, 1.0], data=data, save_conf=True, save_txt=True)
    # demo_base.model_val(val_name, filter_small=0.05, conf=conf, risk_enlarge=1.5)
    # demo_base.model_val(val_name, filter_small=0.05, conf=conf, risk_enlarge=1.6)
    # demo_base.model_val(val_name, filter_small=0.05, conf=conf, risk_enlarge=1.7)                                                  # val_name = 'fusedata7961_mseg_c5_l2_1023_src_80p_ref-[yolov10x-mseg-dlka3res-7]3'
    # demo_base.model_val(val_name, filter_small=0.05, conf=conf, risk_enlarge=1.8)                                                  # data = 'defect_test_1023.yaml'
    # demo_base.model_val(val_name, filter_small=0.05, conf=conf, risk_enlarge=1.9)                                                  # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, risk_enlarge=[1.0, 1.3, 2.0, 1.0], data=data, save_conf=True, save_txt=True)
    # demo_base.model_val(val_name, filter_small=0.05, conf=conf, risk_enlarge=2.0)                                                  # val_name = 'fusedata7961_mseg_c5_l2_1023_src_80p_ref-[yolov10x-mseg-dlka3res-7]3'
    # demo_base.model_val(val_name, filter_small=0.05, conf=conf, risk_enlarge=3.0)                                                  # val_name = 'fusedata7961_mseg_c5_l2_1023_src_80p_ref-[yolov10x-mseg-dlka3res-7]3'
    # demo_base.model_val(val_name, filter_small=0.05, conf=conf, risk_enlarge=4.0)
    # demo_base.model_val(val_name, filter_small=0.05, conf=conf, risk_enlarge=5.0)
    # demo_base.model_val(val_name, filter_small=0.05, conf=conf, risk_enlarge=6.0)
    # demo_base.model_val(val_name, filter_small=0.05, conf=conf, risk_enlarge=10.0)
    # demo_base.model_val(val_name, filter_small=0.05, conf=conf, risk_enlarge=[10, 1.0, 2.0, 1.0], save_conf=True, save_txt=True)  # data = 'defect_test_1023.yaml'
    # #                                                                                                                                # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, risk_enlarge=[1.0, 1.3, 2.0, 1.0], data=data, save_conf=True, save_txt=True, eval_att_by_class=False)

    # demo_base.model_val(val_name, filter_small=0.05, conf=conf, risk_enlarge=[10, 1.0, 2.0, 1.0], save_conf=True,
    #                     # data = 'fusedata7961_mseg_c5_l2_1029_abandonment_refine_all.yaml',
    #                     save_txt=True)  # data = 'defect_test_1023.yaml'

    # val_name = 'fusedata7961_mseg_c5_l2_1023_src_80p_ref-[yolov10x-mseg-dlka3res-7]3'
    # data = 'fusedata7961_mseg_c5_l2_1023_src_80p_ref.yaml'
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, risk_enlarge=[1.0, 1.3, 2.0, 1.0], data=data, save_conf=True, save_txt=True)

    # val_name = 'fusedata7961_mseg_c5_l2_1023_80p_ref-[yolov10x-mseg-dlka3res-7]3'
    # data = 'defect_test_1023.yaml'
    # demo_base.model_val(val_name, filter_small=0.05, data=data)
    # val_name = 'fusedata7961_mseg_c5_l2_1023_80p_ref-[yolov10x-mseg-dlka3res-7-unet]'
    # data = 'fusedata7961_mseg_c5_l2_1023_src_80p_ref.yaml'
    # demo_base.model_val(val_name, data=data, save_conf=True, save_txt=True)
    # val_name = 'fusedata7961_mseg_c5_l2_1023_80p_ref-[yolov10x-mseg-dlka3res-7-unet]'
    # data = 'defect_test_1023.yaml'
    # demo_base.model_val(val_name, data=data, save_conf=True, save_txt=True)

    # val_name = 'fusedata7961_mseg_c5_l2_1023_src_80p_ref-[yolov10x-mseg-dlka3res-7]3'
    # data = 'fusedata7961_mseg_c5_l2_1023_src_80p_ref.yaml'
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, risk_enlarge=[1.0, 1.3, 2.0, 1.0], data=data, save_conf=True, save_txt=True)
    # val_name = 'fusedata7961_mseg_c5_l2_1023_80p_ref-[yolov10x-mseg-dlka3res-7]3'
    # demo_base.model_val(val_name, save_conf=True, save_txt=True)


    # img_dir = r'/localnvme/data/added_data/test_data1121/images'
    # save_dir = r'/localnvme/data/added_data/test_data1121/images_infer'
    # val_name = 'fusedata7961_mseg_c5_l2_1117_v19_sb_test_broken_syn_v1-[yolov10x-mseg-dlka3res-7-unet-single]'
    # demo_base.model_predict(val_name, img_dir = img_dir, name=save_dir, conf=0.1)
    # demo_base.model_predict(val_name, img_dir = img_dir, name=save_dir, conf=0.4)
    # val_name = 'fusedata7961_mseg_c5_l2_1117_v21_sb_test_broken_syn_v3-[yolov10x-mseg-dlka3res-7-unet-single-texture]2'
    # demo_base.model_predict(val_name, img_dir = img_dir, name=save_dir, conf=0.1)
    # demo_base.model_predict(val_name, img_dir = img_dir, name=save_dir, conf=0.4)

    # demo_base.model_predict(val_name, img_dir = img_dir,name=save_dir, conf=0.2)
    # demo_base.model_predict(val_name, img_dir = img_dir,name=save_dir, conf=0.3)
    # demo_base.model_predict(val_name, img_dir = img_dir,name=save_dir, conf=0.4)
    # demo_base.model_predict(val_name, img_dir = img_dir,name=save_dir, conf=0.5)


    # val_name = 'fusedata7961_mseg_c5_l2_1023_src_80p_ref-[yolov10x-mseg-dlka3res-7]3'
    # # img_dir = r'/localnvme/data/added_data/test_data/test_data_mseg_c6_1021_broken_refine/images'
    # # save_dir = r'/localnvme/data/added_data/test_data/test_data_mseg_c6_1021_broken_refine/predicts/infer'
    # img_dir = r'/localnvme/data/billboard/fused_data/data7961_mseg_c5_l2_1023_src/val_80p_ref/images'
    # save_dir = r'/localnvme/data/billboard/fused_data/data7961_mseg_c5_l2_1023_src/val_80p_ref/images_infer'
    # demo_base.model_predict(val_name, img_dir = img_dir, conf=0.001, name=save_dir,)


    # demo_base.model_val('fusedata7961_mseg_c5_l2_1023_80p_ref-[yolov10x-mseg-dlka3res-7-unet]',
    #                     data='fusedata7961_mseg_c5_l2_1023_src_all.yaml', save_txt=True, save_conf=True)
    # demo_base.model_val('fusedata7961_mseg_c5_l2_1023_80p_ref-[yolov10x-mseg-dlka3res-7-unet]', save_txt=True, save_conf=True, eval_att_by_class=False)

    # demo_base.model_export(r'fusedata7961_mseg_c5_l2_1023_src_80p_ref-[yolov10x-mseg-dlka3res-7]3',
    #                         imgsz=(608,960),batch=6)
    # demo_base.model_export(r'fusedata7961_mseg_c5_l2_1023_80p_ref-[yolov10x-mseg-dlka3res-7]3',
    #                         imgsz=(608,960),batch=6)

    val_name = r'fusedata7961_mseg_c5_l2_1113_v17_test-[yolov10x-mseg-dlka3res-7-unet-sep]2'
    image_dir = r'/localnvme/data/billboard/all_data/mseg_c5_l2/data7961_mseg_c5_l2_1113_v17/val_80p/images'
    # demo_base.model_val(val_name, data='fusedata7961_mseg_c5_l2_1113_v17_test.yaml', save_txt=True, save_conf=True, conf=0.5)
    # demo_base.model_val(val_name, data='fusedata7961_mseg_c5_l2_1113_v17_val.yaml', save_txt=True, save_conf=True, conf=0.001)
    # demo_base.model_val(val_name, data='fusedata7961_mseg_c5_l2_1113_v17_val.yaml', save_txt=True, save_conf=True, conf=0.01)
    # demo_base.model_val(val_name, data='fusedata7961_mseg_c5_l2_1113_v17_val.yaml', save_txt=True, save_conf=True, conf=0.1,filter_small=0.05)
    # demo_base.model_val(val_name, data='fusedata7961_mseg_c5_l2_1113_v17_val.yaml', save_txt=True, save_conf=True, conf=0.2)
    # demo_base.model_val(val_name, data='fusedata7961_mseg_c5_l2_1113_v17_val.yaml', save_txt=True, save_conf=True, conf=0.3)
    # demo_base.model_val(val_name, data='fusedata7961_mseg_c5_l2_1113_v17_val.yaml', save_txt=True, save_conf=True, conf=0.4)
    # demo_base.model_val(val_name, data='fusedata7961_mseg_c5_l2_1113_v17_val.yaml', save_txt=True, save_conf=True, conf=0.5)

    # demo_base.model_predict(val_name, img_dir=image_dir, conf=0.1, name=image_dir+'_infer', save_conf=True)
    brightness_list = [
        0.5,
        0.75,
        1.25,
        1.5,
    ]
    for brightness in brightness_list:
        image_bright_dir = image_dir + f'_b{int(brightness*100)}'
        demo_base.model_predict(val_name, img_dir=image_bright_dir, conf=0.1, name=image_bright_dir+'_infer', save_conf=True)

    # demo_base.model_export(val_name, imgsz=(608,992),batch=1)
