import demo_mdet
import torch
demo_mdet.DEVICE = torch.device('cuda:1')
demo_mdet.EPOCHS = 100
if __name__ == '__main__':
    pass
    # demo_mdet.myolo10('yolov10x-mdetect.yaml', 'yolov10x.pt', name='exp_mloss_enlarge', mloss_enlarge=0.25)
    # demo_mdet.myolo10('yolov10x-mdetect.yaml', 'yolov10x.pt', name='exp_mloss_enlarge', mloss_enlarge=0.30)
    # demo_mdet.myolo10('yolov10x-mdetect.yaml', 'yolov10x.pt', name='exp_mloss_enlarge', mloss_enlarge=0.35)
    # demo_mdet.myolo10('yolov10x-mdetect.yaml', 'yolov10x.pt', name='exp_mloss_enlarge', mloss_enlarge=0.40)
    # demo_mdet.myolo10('yolov10x-mdetect.yaml', 'yolov10x.pt', name='exp_mloss_enlarge', mloss_enlarge=0.45)
    #
    # demo_mdet.myolo10('yolov10n-mdetect.yaml', 'yolov10n.pt', name='exp_yolo10n', mloss_enlarge=0.3)
    # demo_mdet.myolo10('yolov10s-mdetect.yaml', 'yolov10s.pt', name='exp_yolo10s', mloss_enlarge=0.3)
    # demo_mdet.myolo10('yolov10m-mdetect.yaml', 'yolov10m.pt', name='exp_yolo10m', mloss_enlarge=0.3)
    # demo_mdet.myolo10('yolov10b-mdetect.yaml', 'yolov10b.pt', name='exp_yolo10b', mloss_enlarge=0.3)
    # demo_mdet.myolo10('yolov10l-mdetect.yaml', 'yolov10l.pt', name='exp_yolo10l', mloss_enlarge=0.3)
    # demo_mdet.myolo10('yolov10x-mdetect.yaml', 'yolov10x.pt', name='exp_yolo10x', mloss_enlarge=0.3)

    # demo_mdet.myolo9('yolov9c-mdetect.yaml', 'yolov9c.pt', name='exp_yolo9c', mloss_enlarge=0.3)
    # demo_mdet.myolo9('yolov9e-mdetect.yaml', 'yolov9e.pt', name='exp_yolo9e', mloss_enlarge=0.3)

    
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3tr_3.yaml',  'yolov10x.pt',  name='exp_yolo10x_m3', mloss_enlarge=0.3)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3str_3.yaml',  'yolov10x.pt',  name='exp_yolo10x_m3', mloss_enlarge=0.3)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strsp_3.yaml',  'yolov10x.pt',  name='exp_yolo10x_m3', mloss_enlarge=0.3)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3.yaml',  'yolov10x.pt',  name='exp_yolo10x_m3', mloss_enlarge=0.3)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c4str_3.yaml',  'yolov10x.pt',  name='exp_yolo10x_m3', mloss_enlarge=0.3)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c4str2_3.yaml',  'yolov10x.pt',  name='exp_yolo10x_m3', mloss_enlarge=0.3)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c4str3_3.yaml',  'yolov10x.pt',  name='exp_yolo10x_m3', mloss_enlarge=0.3)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c4strcbam_3.yaml',  'yolov10x.pt',  name='exp_yolo10x_m3', mloss_enlarge=0.3)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c4strcbam2_3.yaml',  'yolov10x.pt',  name='exp_yolo10x_m3', mloss_enlarge=0.3)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c4strcbam3_3.yaml',  'yolov10x.pt',  name='exp_yolo10x_m3', mloss_enlarge=0.3)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3str2_3.yaml', 'yolov10x.pt',  name='exp_yolo10x_m3', mloss_enlarge=0.3)

    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3tr_3_res.yaml',  'yolov10x.pt',  name='exp_yolo10x_m3', mloss_enlarge=0.3)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3str_3_res.yaml',  'yolov10x.pt',  name='exp_yolo10x_m3', mloss_enlarge=0.3)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strsp_3_res.yaml',  'yolov10x.pt',  name='exp_yolo10x_m3', mloss_enlarge=0.3)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_res.yaml',  'yolov10x.pt',  name='exp_yolo10x_m3', mloss_enlarge=0.3)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c4str_3_res.yaml',  'yolov10x.pt',  name='exp_yolo10x_m3', mloss_enlarge=0.3)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c4str2_3_res.yaml',  'yolov10x.pt',  name='exp_yolo10x_m3', mloss_enlarge=0.3)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c4str3_3_res.yaml',  'yolov10x.pt',  name='exp_yolo10x_m3', mloss_enlarge=0.3)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c4strcbam_3_res.yaml',  'yolov10x.pt',  name='exp_yolo10x_m3', mloss_enlarge=0.3)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c4strcbam2_3_res.yaml',  'yolov10x.pt',  name='exp_yolo10x_m3', mloss_enlarge=0.3)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c4strcbam3_3_res.yaml',  'yolov10x.pt',  name='exp_yolo10x_m3', mloss_enlarge=0.3)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3str2_3_res.yaml', 'yolov10x.pt',  name='exp_yolo10x_m3', mloss_enlarge=0.3)

    # demo_mdet.myolo9('yolov9m-mdetect.yaml', 'yolov9m.pt', name='exp_yolo9m', mloss_enlarge=0.3)
    # demo_mdet.myolo9('yolov9s-mdetect.yaml', 'yolov9s.pt', name='exp_yolo9s', mloss_enlarge=0.3)
    # demo_mdet.myolo9('yolov9c-mdetect.yaml', 'yolov9c.pt', name='exp_yolo9c', mloss_enlarge=0.3)
    # demo_mdet.myolo9('yolov9m-mdetect.yaml', 'yolov9m.pt', name='exp_yolo9m', mloss_enlarge=0.3, seed=1)
    # demo_mdet.myolo9('yolov9s-mdetect.yaml', 'yolov9s.pt', name='exp_yolo9s', mloss_enlarge=0.3, seed=1)
    # demo_mdet.myolo9('yolov9c-mdetect.yaml', 'yolov9c.pt', name='exp_yolo9c', mloss_enlarge=0.3, seed=1)
    # demo_mdet.myolo9('yolov9m-mdetect.yaml', 'yolov9m.pt', name='exp_yolo9m', mloss_enlarge=0.3, seed=100000)
    # demo_mdet.myolo9('yolov9s-mdetect.yaml', 'yolov9s.pt', name='exp_yolo9s', mloss_enlarge=0.3, seed=100000)
    # demo_mdet.myolo9('yolov9c-mdetect.yaml', 'yolov9c.pt', name='exp_yolo9c', mloss_enlarge=0.3, seed=100000)

    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt', name='exp_yolo10x_mmm',
    #         mloss_enlarge=0.3, retrain=True, fine_tune=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head32.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt', name='exp_yolo10x_mmm',
    #         mloss_enlarge=0.3, retrain=True, fine_tune=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head100.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt', name='exp_yolo10x_mmm',
    #         mloss_enlarge=0.3, retrain=True, fine_tune=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep2head32.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt', name='exp_yolo10x_mmm',
    #         mloss_enlarge=0.3, retrain=True, fine_tune=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep2head100.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt', name='exp_yolo10x_mmm',
    #         mloss_enlarge=0.3, retrain=True, fine_tune=True)

    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt', name='exp_yolo10x_mmm',
    #         mloss_enlarge=0.3, retrain=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_retrain.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt', name='exp_yolo10x_mmm',
    #         mloss_enlarge=0.3, retrain=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head32.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt', name='exp_yolo10x_mmm',
    #         mloss_enlarge=0.3, retrain=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head100.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt', name='exp_yolo10x_mmm',
    #         mloss_enlarge=0.3, retrain=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep2head32.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt', name='exp_yolo10x_mmm',
    #         mloss_enlarge=0.3, retrain=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep2head100.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt', name='exp_yolo10x_mmm',
    #         mloss_enlarge=0.3, retrain=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep3head.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt', name='exp_yolo10x_mmm',
    #         mloss_enlarge=0.3, retrain=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep4head.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt', name='exp_yolo10x_mmm',
    #         mloss_enlarge=0.3, retrain=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep5head.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt', name='exp_yolo10x_mmm',
    #         mloss_enlarge=0.3, retrain=True)



    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom6.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='exp_yolo10x_head', mloss_enlarge=0.3, retrain=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom6_res.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='exp_yolo10x_head', mloss_enlarge=0.3, retrain=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom6_nosf.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='exp_yolo10x_head', mloss_enlarge=0.3, retrain=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom6_res_nosf.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='exp_yolo10x_head', mloss_enlarge=0.3, retrain=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom6_pure.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='exp_yolo10x_head', mloss_enlarge=0.3, retrain=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom6_res_pure.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='exp_yolo10x_head', mloss_enlarge=0.3, retrain=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom6_nosf_pure.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='exp_yolo10x_head', mloss_enlarge=0.3, retrain=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom6_res_nosf_pure.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='exp_yolo10x_head', mloss_enlarge=0.3, retrain=True)

    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom1.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='exp_yolo10x_head', mloss_enlarge=0.3, retrain=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom1_res.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='exp_yolo10x_head', mloss_enlarge=0.3, retrain=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom1_nosf.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='exp_yolo10x_head', mloss_enlarge=0.3, retrain=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom1_res_nosf.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='exp_yolo10x_head', mloss_enlarge=0.3, retrain=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom1_pure.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='exp_yolo10x_head', mloss_enlarge=0.3, retrain=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom1_res_pure.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='exp_yolo10x_head', mloss_enlarge=0.3, retrain=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom1_nosf_pure.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='exp_yolo10x_head', mloss_enlarge=0.3, retrain=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom1_res_nosf_pure.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='exp_yolo10x_head', mloss_enlarge=0.3, retrain=True)
    #
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom2.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='exp_yolo10x_head', mloss_enlarge=0.3, retrain=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom2_res.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='exp_yolo10x_head', mloss_enlarge=0.3, retrain=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom2_nosf.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='exp_yolo10x_head', mloss_enlarge=0.3, retrain=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom2_res_nosf.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='exp_yolo10x_head', mloss_enlarge=0.3, retrain=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom2_pure.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='exp_yolo10x_head', mloss_enlarge=0.3, retrain=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom2_res_pure.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='exp_yolo10x_head', mloss_enlarge=0.3, retrain=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom2_nosf_pure.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='exp_yolo10x_head', mloss_enlarge=0.3, retrain=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom2_res_nosf_pure.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='exp_yolo10x_head', mloss_enlarge=0.3, retrain=True)
    #
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom3.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='exp_yolo10x_head', mloss_enlarge=0.3, retrain=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom3_res.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='exp_yolo10x_head', mloss_enlarge=0.3, retrain=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom3_nosf.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='exp_yolo10x_head', mloss_enlarge=0.3, retrain=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom3_res_nosf.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='exp_yolo10x_head', mloss_enlarge=0.3, retrain=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom3_pure.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='exp_yolo10x_head', mloss_enlarge=0.3, retrain=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom3_res_pure.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='exp_yolo10x_head', mloss_enlarge=0.3, retrain=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom3_nosf_pure.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='exp_yolo10x_head', mloss_enlarge=0.3, retrain=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom3_res_nosf_pure.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='exp_yolo10x_head', mloss_enlarge=0.3, retrain=True)
    #
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom4.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='exp_yolo10x_head', mloss_enlarge=0.3, retrain=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom4_res.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='exp_yolo10x_head', mloss_enlarge=0.3, retrain=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom4_nosf.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='exp_yolo10x_head', mloss_enlarge=0.3, retrain=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom4_res_nosf.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='exp_yolo10x_head', mloss_enlarge=0.3, retrain=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom4_pure.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='exp_yolo10x_head', mloss_enlarge=0.3, retrain=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom4_res_pure.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='exp_yolo10x_head', mloss_enlarge=0.3, retrain=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom4_nosf_pure.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='exp_yolo10x_head', mloss_enlarge=0.3, retrain=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom4_res_nosf_pure.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='exp_yolo10x_head', mloss_enlarge=0.3, retrain=True)
    #
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom4.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='exp_yolo10x_head', mloss_enlarge=0.3, retrain=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom4_res.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='exp_yolo10x_head', mloss_enlarge=0.3, retrain=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom4_nosf.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='exp_yolo10x_head', mloss_enlarge=0.3, retrain=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom4_res_nosf.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='exp_yolo10x_head', mloss_enlarge=0.3, retrain=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom4_pure.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='exp_yolo10x_head', mloss_enlarge=0.3, retrain=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom4_res_pure.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='exp_yolo10x_head', mloss_enlarge=0.3, retrain=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom4_nosf_pure.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='exp_yolo10x_head', mloss_enlarge=0.3, retrain=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom4_res_nosf_pure.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='exp_yolo10x_head', mloss_enlarge=0.3, retrain=True)


    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatmlp.yaml',
    #                   'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='exp_yolo10x_head_mc', mloss_enlarge=0.3, retrain=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcos.yaml',
    #                   'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='exp_yolo10x_head_mc', mloss_enlarge=0.3, retrain=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatmlp_res.yaml',
    #                   'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='exp_yolo10x_head_mc', mloss_enlarge=0.3, retrain=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcos_res.yaml',
    #                   'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='exp_yolo10x_head_mc', mloss_enlarge=0.3, retrain=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatmlpt.yaml',
    #                   'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='exp_yolo10x_head_mc', mloss_enlarge=0.3, retrain=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcost.yaml',
    #                   'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='exp_yolo10x_head_mc', mloss_enlarge=0.3, retrain=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatmlpt_res.yaml',
    #                   'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='exp_yolo10x_head_mc', mloss_enlarge=0.3, retrain=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcost_res.yaml',
    #                   'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='exp_yolo10x_head_mc', mloss_enlarge=0.3, retrain=True)


    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatmlpr.yaml',
    #                   'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='exp_yolo10x_head_mc', mloss_enlarge=0.3, retrain=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatmlpr_res.yaml',
    #                   'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='exp_yolo10x_head_mc', mloss_enlarge=0.3, retrain=True)
    #
    #
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_gatcos_res.yaml',
    #                   'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='exp_yolo10x_head_mc', mloss_enlarge=0.3, retrain=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_gatmlpt_res.yaml',
    #                   'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='exp_yolo10x_head_mc', mloss_enlarge=0.3, retrain=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_gatcos_res_retrain.yaml',
    #                   'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='exp_yolo10x_head_mc', mloss_enlarge=0.3, retrain=True)
    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_gatmlpt_res_retrain.yaml',
    #                   'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='exp_yolo10x_head_mc', mloss_enlarge=0.3, retrain=True)

    # demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom44.yaml',
    #                   'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #                   name='debug', mloss_enlarge=0.3, retrain=True)

    demo_mdet.myolo10('yolov10x-mdetect-psa_c3strcp_3_sep6head.yaml',
                      'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
                      name='exp_yolo10x_head_mc', mloss_enlarge=0.3, retrain=True)