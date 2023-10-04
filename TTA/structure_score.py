import os
import json
import cv2
import re
import numpy as np
import pandas as pd
import networkx as nx

def read_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f_read:
        json_read = json.load(f_read)

    return json_read


def write_json(json_path, json_object):
    with open(json_path, 'w', encoding='utf-8') as f_write:
        json.dump(json_object, f_write)


def computer_wide_overlap(cell_1_x_min, cell_2_x_min, cell_1_x_max, cell_2_x_max):
    overlap_x_min = max(cell_1_x_min, cell_2_x_min)
    overlap_x_max = min(cell_1_x_max, cell_2_x_max)
    overlap_wide = overlap_x_max - overlap_x_min
    cell_1_wide = cell_1_x_max - cell_1_x_min
    cell_2_wide = cell_2_x_max - cell_2_x_min
    overlap_wide_rate_1 = overlap_wide / cell_1_wide
    overlap_wide_rate_2 = overlap_wide / cell_2_wide
    if overlap_wide_rate_1 >= 0.3:
        return 1
    else:
        if overlap_wide_rate_2 >= 0.5:
            return 1
        else:
            return 0


def computer_height_overlap(cell_1_y_min, cell_2_y_min, cell_1_y_max, cell_2_y_max):
    overlap_y_min = max(cell_1_y_min, cell_2_y_min)
    overlap_y_max = min(cell_1_y_max, cell_2_y_max)
    overlap_height = overlap_y_max - overlap_y_min
    cell_1_height = cell_1_y_max - cell_1_y_min
    cell_2_height = cell_2_y_max - cell_2_y_min
    overlap_height_rate_1 = overlap_height / cell_1_height
    overlap_height_rate_2 = overlap_height / cell_2_height
    if overlap_height_rate_1 >= 0.3:
        return 1
    else:
        if overlap_height_rate_2 >= 0.5:
            return 1
        else:
            return 0


def computer_up_down_left_right(cell_1, cell_2):
    cell_1_x_min = cell_1[0]
    cell_1_y_min = cell_1[1]
    cell_1_x_max = cell_1[2]
    cell_1_y_max = cell_1[3]

    cell_2_x_min = cell_2[0]
    cell_2_y_min = cell_2[1]
    cell_2_x_max = cell_2[2]
    cell_2_y_max = cell_2[3]
    # up
    if cell_1_y_min > cell_2_y_max:
        wide_overlap_flag = computer_wide_overlap(cell_1_x_min, cell_2_x_min, cell_1_x_max, cell_2_x_max)
        if wide_overlap_flag:
            return 'up'
        else:
            return 'no'

    # down
    if cell_1_y_max <= cell_2_y_min:
        wide_overlap_flag = computer_wide_overlap(cell_1_x_min, cell_2_x_min, cell_1_x_max, cell_2_x_max)
        if wide_overlap_flag:
            return 'down'
        else:
            return 'no'

    # left
    if cell_1_x_min > cell_2_x_max:
        height_overlap_flag = computer_height_overlap(cell_1_y_min, cell_2_y_min, cell_1_y_max, cell_2_y_max)
        if height_overlap_flag:
            return 'left'
        else:
            return 'no'

    # right
    if cell_1_x_max <= cell_2_x_min:
        height_overlap_flag = computer_height_overlap(cell_1_y_min, cell_2_y_min, cell_1_y_max, cell_2_y_max)
        if height_overlap_flag:
            return 'right'
        else:
            return 'no'


def remove_cell_in_gap(same_direction_res_dict, json_res, i):
    for k, v in same_direction_res_dict.items():
        if k == 'up':
            if len(v) == 1:
                pass
            else:
                for z in range(len(v) - 1, -1, -1):
                    gap_y_min = v[z][3]
                    gap_y_max = json_res[i]['bbox'][1]
                    gap_x_min = v[z][0]
                    gap_x_max = v[z][2]
                    gap_x_wide = gap_x_max - gap_x_min
                    for w in range(len(v)):
                        if w != z:
                            w_y_min = v[w][1]
                            w_y_max = v[w][3]
                            w_x_min = v[w][0]
                            w_x_max = v[w][2]
                            overlap_x_min = max(gap_x_min, w_x_min)
                            overlap_x_max = min(gap_x_max, w_x_max)
                            middle_cell_wide = w_x_max - w_x_min
                            overlap_x_distance = overlap_x_max - overlap_x_min
                            overlap_x_rate = overlap_x_distance / middle_cell_wide
                            if w_y_min >= gap_y_min and w_y_max <= gap_y_max and  overlap_x_rate > 0:
                                v.pop(z)
                                break
        if k == 'down':
            if len(v) == 1:
                pass
            else:
                for z in range(len(v) - 1, -1, -1):
                    gap_y_min = json_res[i]['bbox'][3]
                    gap_y_max = v[z][1]

                    gap_x_min = v[z][0]
                    gap_x_max = v[z][2]
                    gap_x_wide = gap_x_max - gap_x_min
                    for w in range(len(v)):
                        if w != z:
                            w_y_min = v[w][1]
                            w_y_max = v[w][3]

                            w_x_min = v[w][0]
                            w_x_max = v[w][2]
                            overlap_x_min = max(gap_x_min, w_x_min)
                            overlap_x_max = min(gap_x_max, w_x_max)
                            middle_cell_wide = w_x_max - w_x_min
                            overlap_x_distance = overlap_x_max - overlap_x_min
                            overlap_x_rate = overlap_x_distance / middle_cell_wide

                            if w_y_min >= gap_y_min and w_y_max <= gap_y_max and overlap_x_rate > 0:
                                v.pop(z)
                                break

        if k == 'left':
            if len(v) == 1:
                pass
            else:
                for z in range(len(v) - 1, -1, -1):
                    gap_x_min = v[z][2]
                    gap_x_max = json_res[i]['bbox'][0]

                    gap_y_min = v[z][1]
                    gap_y_max = v[z][3]
                    gap_y_wide = gap_y_max - gap_y_min

                    for w in range(len(v)):
                        if w != z:
                            w_x_min = v[w][0]
                            w_x_max = v[w][2]

                            w_y_min = v[w][1]
                            w_y_max = v[w][3]
                            overlap_y_min = max(gap_y_min, w_y_min)
                            overlap_y_max = min(gap_y_max, w_y_max)
                            middle_cell_wide = w_y_max - w_y_min
                            overlap_y_distance = overlap_y_max - overlap_y_min
                            overlap_y_rate = overlap_y_distance / middle_cell_wide
                            if w_x_min >= gap_x_min and w_x_max <= gap_x_max and overlap_y_rate > 0:
                                v.pop(z)
                                break

        if k == 'right':
            if len(v) == 1:
                pass
            else:
                for z in range(len(v) - 1, -1, -1):
                    gap_x_min = json_res[i]['bbox'][0]
                    gap_x_max = v[z][2]

                    gap_y_min = v[z][1]
                    gap_y_max = v[z][3]
                    gap_y_wide = gap_y_max - gap_y_min
                    for w in range(len(v)):
                        if w != z:
                            w_x_min = v[w][0]
                            w_x_max = v[w][2]

                            w_y_min = v[w][1]
                            w_y_max = v[w][3]
                            overlap_y_min = max(gap_y_min, w_y_min)
                            overlap_y_max = min(gap_y_max, w_y_max)
                            middle_cell_wide = w_y_max - w_y_min
                            overlap_y_distance = overlap_y_max - overlap_y_min
                            overlap_y_rate = overlap_y_distance / middle_cell_wide
                            if w_x_min >= gap_x_min and w_x_max <= gap_x_max and overlap_y_rate > 0:
                                v.pop(z)
                                break

    return same_direction_res_dict


def draw_bbox(same_direction_res_dict, base_cell, image_path, output_dir, i):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    import re
    image = cv2.imread(image_path)
    image_name = re.split('/', image_path)[-1]
    image_name = image_name[:-4]+"_"+str(i)+image_name[-4:]
    same_direction_res_dict['base'] = [base_cell]
    for k,bbox_list in same_direction_res_dict.items():
        for bbox in bbox_list:
            xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
            image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            if len(bbox) == 4:
                import random
                y_text = int(random.uniform(ymin, ymax))
                cv2.putText(image, str(k), (xmax, y_text), 2, 1, (255, 255, 0))
    # print("os.path.join(output_dir, image_name)：", os.path.join(output_dir, image_name))
    cv2.imwrite(os.path.join(output_dir, image_name), image)


def draw_structure(c_root, same_direction_res_dict, file, base_cell, i):
    output_image_only_cell_test = os.path.join(c_root, 'output_image_only_cell_test')
    draw_bbox(same_direction_res_dict, base_cell, os.path.join(output_image_only_cell_test, file.replace('json', 'jpg')),
              output_image_only_cell_test + "_structure_draw", i)

    # for file in os.listdir(ground_truth):
    #     file_path = os.path.join(ground_truth, file)
    #     gt = read_json(file_path)
    #     bbox = [item['bbox'] for item in gt]
    #     draw_predict(bbox, os.path.join(output_image_only_cell_test, file.replace('json','jpg')), output_image_only_cell_test+"_gt_draw")



def structure_score(file_path):
    sources = []
    targets = []
    weights = []
    # print("file_path____:", file_path)
    # if '111_3' in file_path:
    #     print("  ")
    with open(file_path, 'r', encoding='utf8') as f_read:
        for line in f_read:
            line_split = re.split('%%%%', line.strip())
            if len(line_split) == 2:
                sources.append(line_split[0])
                targets.append(line_split[1])
                weights.append(1)

    edge = pd.DataFrame()
    edge['sources'] = sources
    edge['targets'] = targets
    edge['weights'] = weights

    G = nx.from_pandas_edgelist(edge, source='sources', target='targets',edge_attr='weights')
    print("degree")
    print(nx.degree(G))
    degree = nx.degree(G)
    # 联通分量
    print(list(nx.connected_components(G)))
    connected_components = list(nx.connected_components(G))
    # 图径径
    # print(nx.diameter(G))
    # 度中心性
    print(nx.degree_centrality(G))
    degree_centrality = nx.degree_centrality(G)
    # 特征向量中心性
    try:
        print(nx.eigenvector_centrality(G))
        eigenvector_centrality = nx.eigenvector_centrality(G)
    except:
        eigenvector_centrality = {}
        print('error')
    # between
    print(nx.betweenness_centrality(G))
    betweenness_centrality = nx.betweenness_centrality(G)
    # closeness
    print(nx.closeness_centrality(G))
    closeness_centrality = nx.closeness_centrality(G)
    # pagerank
    print(nx.pagerank(G))
    pagerank = nx.pagerank(G)
    # HITs
    print(nx.hits(G))
    hubs = nx.hits(G)[0]
    authorities = nx.hits(G)[1]

    degree_compare = []
    for cell_degree in degree:
        k,degree = cell_degree
        cell = k
        degree_compare.append({'cell':k,
         'degree':degree,
         # 'connected_components': connected_components[k],
         'degree_centrality': degree_centrality[k],
         'eigenvector_centrality': eigenvector_centrality.get(k, 'None'),
         'betweenness_centrality': betweenness_centrality[k],
         'closeness_centrality': closeness_centrality[k],
         'pagerank': pagerank[k],
         'hubs': hubs[k],
         'authorities': authorities[k],
         })

    file_path = file_path.replace('structure_score', 'structure_score_collect')
    file_path = file_path.replace('txt','xlsx')
    degree_compare_df = pd.DataFrame(degree_compare)
    degree_compare_df.to_excel(file_path)

def computer_structure_score(cell_res_dir, c_root):
    for file in os.listdir(cell_res_dir):
        print("file:", file)
        file_path = os.path.join(cell_res_dir, file)
        json_res = read_json(file_path)
        # print(json_res)
        write_path = os.path.join(c_root+'structure_score', file.replace('json', 'txt'))
        f_write = open(write_path, 'w', encoding='utf8')

        for i in range(len(json_res)):
            same_direction_res_dict = {}
            for j in range(len(json_res)):
                if i != j:
                    direction = computer_up_down_left_right(json_res[i]['bbox'], json_res[j]['bbox'])
                    if direction in ['up', 'down', 'right', 'left']:
                        if direction not in same_direction_res_dict:
                            same_direction_res_dict[direction] = [json_res[j]['bbox']]
                        else:
                            same_direction_res_dict[direction].append(json_res[j]['bbox'])

            same_direction_res_dict = remove_cell_in_gap(same_direction_res_dict, json_res, i)
            for k, v in same_direction_res_dict.items():
                if len(v) >= 3:
                    print('file:', file)
                    print("i:", i)
                    print("json_res_i:", json_res[i])
                    print("same_direction_res_dict:", same_direction_res_dict)
                    print('same_direction_res_dict filter：', same_direction_res_dict)
                    print("\n\n")
                    # continue

                for vv in v:
                    f_write.write(str(json_res[i]['bbox']) + '%%%%'+ str(vv) +'\n')

            draw_structure(c_root, same_direction_res_dict, file, json_res[i]['bbox'] , i)

        f_write.close()
        structure_score(write_path)

if __name__ == '__main__':
    c_root = '/data/cs_lzhan011/uq/mmdetection/data/Cell_split_train_test/Cell_images/'
    # cell_res = os.path.join(c_root, 'output_image_only_cell_test_predict')
    cell_res = '/data/cs_lzhan011/uq/mmdetection/data/Cell_split_train_test/coco_split_table_input'
    # c_root = '/home/lei/fsdownload/tta_20221130'
    computer_structure_score(cell_res, c_root)
