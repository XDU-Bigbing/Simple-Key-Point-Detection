import json
import numpy as np
import torch


def nms(bboxes, scores, threshold=0.5):
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (x2-x1)*(y2-y1)   # [N,] 每个bbox的面积
    _, order = scores.sort(0, descending=True)    # 降序排列

    keep = []
    while order.numel() > 0:       # torch.numel()返回张量元素个数
        if order.numel() == 1:     # 保留框只剩一个
            i = order.item()
            keep.append(i)
            break
        else:
            i = order[0].item()    # 保留scores最大的那个框box[i]
            keep.append(i)

        # 计算box[i]与其余各框的IOU(思路很好)
        xx1 = x1[order[1:]].clamp(min=x1[i])   # [N-1,]
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])
        inter = (xx2-xx1).clamp(min=0) * (yy2-yy1).clamp(min=0)   # [N-1,]

        iou = inter / (areas[i]+areas[order[1:]]-inter)  # [N-1,]
        # 注意此时idx为[N-1,] 而order为[N,]
        idx = (iou <= threshold).nonzero().squeeze()
        if idx.numel() == 0:
            break
        order = order[idx+1]  # 修补索引之间的差值
    return keep


def results_integration(result_dict):

    # with open("result.json",'r') as result_f:
    #     result_dict = json.load(result_f)

    result_dict = {item['image_id'].split(
        '/')[1]: item for item in result_dict}

    # print(result_dict)

    with open("../DATA/tile_round1_testA_20201231/info.json", 'r') as info_f:
    # with open("test_info.json", 'r') as info_f:
        info_dict = json.load(info_f)

    # print(info_dict)

    final_result = {}

    for item in info_dict:
        sub_img_name = item['sub_image_name'].split('.')[0]
        img_name = item['image_name']
        offset = item['offset']
        shrink_ratio = item['shrink_ratio']

        boxes = result_dict[sub_img_name]['boxes']
        boxes_num = boxes.shape[0]
        offset_np = np.tile(np.array(offset), (boxes_num, 2))
        shrink_ratio_np = np.tile(np.array(shrink_ratio), (boxes_num, 2))
        boxes_new = (boxes+offset_np)*shrink_ratio_np
        scores = result_dict[sub_img_name]["scores"]
        labels = result_dict[sub_img_name]["labels"]

        if img_name in final_result:
            # print('in')
            final_result[img_name]["boxes"] = np.concatenate(
                (final_result[img_name]["boxes"], boxes_new), axis=0)
            final_result[img_name]["scores"] = np.concatenate(
                (final_result[img_name]["scores"], scores), axis=0)
            final_result[img_name]["labels"] = np.concatenate(
                (final_result[img_name]["labels"], labels), axis=0)
        else:
            # print('not in')
            final_result[img_name] = {"boxes": boxes_new,
                                      "scores": scores,
                                      "labels": labels}
    output = []
    for img_name in final_result.keys():
        # print(img_name)
        boxes = torch.from_numpy(final_result[img_name]["boxes"])
        scores = torch.from_numpy(final_result[img_name]["scores"])
        keep = nms(boxes, scores)
        for index in keep:
            keep_boxes = final_result[img_name]["boxes"][index].tolist()
            keep_boxes = [round(n) for n in keep_boxes]
            output.append({
                "name": img_name,
                "category": final_result[img_name]["labels"][index].tolist(),
                "bbox": keep_boxes,
                "score": final_result[img_name]["scores"][index].tolist()
            })

    with open('output.json', 'w') as fp:
        json.dump(output, fp, indent=4, ensure_ascii=False)
    
