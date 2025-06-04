
import numpy as np
from serve.utils import remove_outliers


def get_scene_graph(image, pcd, mask, object_names):
    if len(mask) == 0:
        return [], []
    n, h, w = mask.shape
    image = np.array(image)

    objects_info = []
    objects_dict = []
    for i in range(n):
        object_mask = mask[i]
        segmented_object = pcd[object_mask]

        segmented_object = remove_outliers(segmented_object)
        min_values = segmented_object.min(axis=0)
        max_values = segmented_object.max(axis=0)
        mean_values = segmented_object.mean(axis=0)
        center = f"x: {mean_values[0]:.2f}, y: {mean_values[1]:.2f}, z: {mean_values[2]:.2f}"
        bbox = {
            "x_min ~ x_max": f"{min_values[0]:.2f} ~ {max_values[0]:.2f}",
            "y_min ~ y_max": f"{min_values[1]:.2f} ~ {max_values[1]:.2f}",
            "z_min ~ z_max": f"{min_values[2]:.2f} ~ {max_values[2]:.2f}"
        }
        node = {
            'id': i + 1,
            'object name': object_names[i],
            'center': center,
            'bounding box': bbox,
        }
        objects_info.append(node)

        node_dict = {
            'object name': object_names[i],
            'center': [round(mean_values[0], 2), round(mean_values[1], 2), round(mean_values[2], 2)],
            'bounding box': [
                [min_values[0], max_values[0]],
                [min_values[1], max_values[1]],
                [min_values[2], max_values[2]]
            ]
        }
        objects_dict.append(node_dict)

    return objects_info, objects_dict
