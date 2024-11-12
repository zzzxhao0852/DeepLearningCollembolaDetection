import os
import json

def check_json_files(directory):
    """
    Check if each annotated JSON file has any annotation errors.

    Args:
        directory (str): The directory path containing images and JSON files.
    """
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".jpg"):
                # Construct the corresponding JSON file path
                json_file = os.path.splitext(file)[0] + ".json"
                json_path = os.path.join(root, json_file)
                print(json_path)

                if not os.path.exists(json_path):
                    print(f"No such JSON file: {json_path}")
                    exit(0)

                with open(json_path, 'r') as f:
                    data = json.load(f)

                    image_height = data.get('imageHeight')
                    image_width = data.get('imageWidth')
                    # Check if the image dimensions are standard: imageHeight=1740, imageWidth=2320
                    if not (image_height == 1740 and image_width == 2320):
                        print(f"Invalid imageHeight or imageWidth in JSON file: {json_path}")
                        exit(0)

                    shapes = data.get('shapes', [])

                    rectangle_count = sum(1 for shape in shapes if
                                          shape['shape_type'] == 'rectangle' and any(
                                              char.isascii() for char in shape['label']))
                    points_count = sum(
                        1 for shape in shapes if shape['shape_type'] == 'point' and shape['label'].strip().isdigit())

                    # Check if there is exactly one rectangle
                    if rectangle_count != 1:
                        print(f"Invalid rectangle in JSON file: {json_path}")
                        exit(0)

                    # Check if there are exactly 4 points
                    if points_count != 4:
                        print(f"Invalid points in JSON file: {json_path}")
                        exit(0)

                    rectangle = [shape for shape in shapes if
                                 shape['shape_type'] == 'rectangle' and any(char.isascii() for char in shape['label'])]
                    points = [shape for shape in shapes if
                              shape['shape_type'] == 'point' and shape['label'].strip().isdigit()]

                    # Check if the rectangle coordinates are from top-left to bottom-right
                    rectangle_points = rectangle[0]['points']
                    if rectangle_points[0][0] > rectangle_points[1][0] or rectangle_points[0][1] > rectangle_points[1][1]:
                        print(f"Invalid rectangle coordinates in file: {json_path}")
                        exit(0)

                    # Check if the point coordinates are within the rectangle
                    rect_x1, rect_y1 = rectangle_points[0]
                    rect_x2, rect_y2 = rectangle_points[1]
                    for point in points:
                        px, py = point['points'][0]
                        if not (rect_x1 <= px <= rect_x2 and rect_y1 <= py <= rect_y2):
                            print(f"Point: {point['label']} coordinates are not within the rectangle in file: {json_path}")
                            exit(0)

                    # Check if point labels are not repeated
                    point_labels = []
                    for point in points:
                        if point['label'] in point_labels:
                            print(f"Point: {point['label']} is repeated with other points in file: {json_path}")
                            exit(0)
                        point_labels.append(point['label'])

if __name__ == "__main__":
    directory = r'../init_dataset'
    check_json_files(directory)