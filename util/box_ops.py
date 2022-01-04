import torch
import cv2

def get_rotated_box(boxes, angle, cx, cy, w, h):
    '''
    cx,cy,w,h : Center of Image, Width and Height
    '''
    corners = get_corners(boxes)
    rotated_corners = rotate_box(corners, angle, cx, cy, w, h)
    rotated_boxes = get_enclosing_box(rotated_corners)

    return rotated_boxes

def rotate_box(corners, angle, cx, cy, w, h):
    corners = corners.reshape(-1,2)
    corners = torch.hstack((corners, torch.ones(corners.shape[0], dtype=corners[0][0].dtype).reshape(-1,1)))

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    M = torch.as_tensor(M,dtype=corners[0][0].dtype)
                        
    cos = torch.abs(M[0,0])
    sin = torch.abs(M[0,1])
                                    
    nW = int((h*sin) + (w*cos))
    nH = int((h*cos) + (w*sin))
                                                
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy
                                                            
    calculated = torch.matmul(M, corners.T).T
    calculated = calculated.reshape(-1,8)
    return calculated

def get_corners(boxes):
    '''
    input
    boxes shape : [n_boxes, 4]
    boxes format: lx,ly,rx,ry

    return 
    boxes shape : [n_boxes, 8]
    boxes format: lx,ly,rx,ly,lx,ry,rx,ry(x1,y1,x2,y2,x3,y3,x4,y4)
    '''

    lx = boxes[:,0].reshape(-1,1)
    ly = boxes[:,1].reshape(-1,1)
    rx = boxes[:,2].reshape(-1,1)
    ry = boxes[:,3].reshape(-1,1)

    return torch.hstack((lx,ly,rx,ly,lx,ry,rx,ry))

def get_enclosing_box(corners):
    x = corners[:,[0,2,4,6]]
    y = corners[:,[1,3,5,7]]
    xmin = torch.min(x,1).values.reshape(-1,1)
    ymin = torch.min(y,1).values.reshape(-1,1)
    xmax = torch.max(x,1).values.reshape(-1,1)
    ymax = torch.max(y,1).values.reshape(-1,1)

    final = torch.hstack((xmin, ymin, xmax, ymax))
    return final

# This function is from https://github.com/kuangliu/pytorch-ssd.
def calc_iou_tensor(box1, box2):
    """ Calculation of IoU based on two boxes tensor,
        Reference to https://github.com/kuangliu/pytorch-src
        input:
            box1 (N, 4)
            box2 (M, 4)
        output:
            IoU (N, M)
    """
    N = box1.size(0)
    M = box2.size(0)

    be1 = box1.unsqueeze(1).expand(-1, M, -1)
    be2 = box2.unsqueeze(0).expand(N, -1, -1)

    # Left Top & Right Bottom
    lt = torch.max(be1[:,:,:2], be2[:,:,:2])
    #mask1 = (be1[:,:, 0] < be2[:,:, 0]) ^ (be1[:,:, 1] < be2[:,:, 1])
    #mask1 = ~mask1
    rb = torch.min(be1[:,:,2:], be2[:,:,2:])
    #mask2 = (be1[:,:, 2] < be2[:,:, 2]) ^ (be1[:,:, 3] < be2[:,:, 3])
    #mask2 = ~mask2

    delta = rb - lt
    delta[delta < 0] = 0
    intersect = delta[:,:,0]*delta[:,:,1]
    #*mask1.float()*mask2.float()

    delta1 = be1[:,:,2:] - be1[:,:,:2]
    area1 = delta1[:,:,0]*delta1[:,:,1]
    delta2 = be2[:,:,2:] - be2[:,:,:2]
    area2 = delta2[:,:,0]*delta2[:,:,1]

    iou = intersect/(area1 + area2 - intersect)
    return iou

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def box_cxcywh_to_ltrb(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

