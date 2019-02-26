import cv2 
def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

# function to draw bounding box on the detected object with class name

def draw_bounding_box(img, classes, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])

    img=img[...,::-1]
    
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), (0,255,0), 2)
    cv2.putText(img, label, (x, y+12), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    
# cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)