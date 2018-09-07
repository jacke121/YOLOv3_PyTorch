import torch
import torch.nn as nn
import numpy as np
import math

from common.utils import bbox_iou

class YOLOLayer(nn.Module):
    def __init__(self, batch_size,layer_num, anchors, num_classes, img_size):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.img_size = img_size

        self.ignore_threshold = 0.9
        self.lambda_xy = 4
        self.lambda_wh = 4
        self.lambda_conf = 4
        self.lambda_cls = 2

        cuda = True if torch.cuda.is_available() else False
        self.FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
        g_dim = 11
        bs = batch_size
        x = [batch_size, 3, g_dim, g_dim]
        if layer_num == 1:
            g_dim *=2
            x = [batch_size, 3, g_dim, g_dim]

        elif layer_num == 2:
            g_dim *= 4
            x = [batch_size, 3, g_dim, g_dim]

        self.grid_x = torch.linspace(0, g_dim - 1, g_dim).repeat(g_dim, 1).repeat(bs * self.num_anchors, 1, 1).view(x).type(self.FloatTensor)
        self.grid_y = torch.linspace(0, g_dim - 1, g_dim).repeat(g_dim, 1).t().repeat(bs * self.num_anchors, 1, 1).view(
            x).type(self.FloatTensor)
        self.stride = self.img_size[0] / g_dim
        self.scaled_anchors = [(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors]
        anchor_w = self.FloatTensor(self.scaled_anchors).index_select(1, self.LongTensor([0]))
        anchor_h = self.FloatTensor(self.scaled_anchors).index_select(1, self.LongTensor([1]))
        self.anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, g_dim * g_dim).view(x)
        self.anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, g_dim * g_dim).view(x)
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        if cuda:
            self.mse_loss = self.mse_loss.cuda()
            self.bce_loss = self.bce_loss.cuda()

        self.conf_mask = torch.zeros(batch_size, self.num_anchors, g_dim, g_dim,requires_grad=False)
        self.tx = torch.zeros(batch_size, self.num_anchors, g_dim, g_dim,requires_grad=False)
        self.ty = torch.zeros(batch_size, self.num_anchors, g_dim, g_dim,requires_grad=False)
        self.tw = torch.zeros(batch_size, self.num_anchors, g_dim, g_dim,requires_grad=False)
        self.th = torch.zeros(batch_size, self.num_anchors, g_dim, g_dim,requires_grad=False)
        self.tconf = torch.zeros(batch_size, self.num_anchors, g_dim, g_dim,requires_grad=False)
        self.tcls = torch.zeros(batch_size, self.num_anchors, g_dim, g_dim, num_classes,requires_grad=False)
        self.noobj_mask = torch.ones(bs, self.num_anchors, g_dim, g_dim, requires_grad=False)

    def forward(self, input, targets=None):
        bs = input.size(0)
        in_h = input.size(2)
        in_w = input.size(3)
        # stride_h = self.img_size[1] / in_h
        # stride_w = self.img_size[0] / in_w
        # scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]

        prediction = input.view(bs,  self.num_anchors,
                                self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])          # Center x
        y = torch.sigmoid(prediction[..., 1])          # Center y
        w = prediction[..., 2]                         # Width
        h = prediction[..., 3]                         # Height
        conf = torch.sigmoid(prediction[..., 4])       # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        pred_boxes = self.FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        if targets is not None:

            #  build target
            nGT, nCorrect = self.get_target(pred_boxes.cpu().data,targets, self.scaled_anchors,
                                                                           in_w, in_h,
                                                                           self.ignore_threshold)
            recall = float(nCorrect / nGT) if nGT else 1
            conf_mask, noobj_mask = self.conf_mask.cuda(), self.noobj_mask.cuda()
            tx, ty, tw, th = self.tx.cuda(), self.ty.cuda(), self.tw.cuda(), self.th.cuda()
            tconf, tcls = self.tconf.cuda(), self.tcls.cuda()
            #  losses.
            loss_x = self.bce_loss(x * conf_mask, tx * conf_mask)
            loss_y = self.bce_loss(y * conf_mask, ty * conf_mask)
            loss_w = self.mse_loss(w * conf_mask, tw * conf_mask)
            loss_h = self.mse_loss(h * conf_mask, th * conf_mask)
            loss_conf = self.bce_loss(conf * conf_mask, conf_mask) + \
                0.5 * self.bce_loss(conf * noobj_mask, noobj_mask * 0.0)
            loss_cls = self.bce_loss(pred_cls[conf_mask == 1], tcls[conf_mask == 1])
            #  total loss = losses * weight
            loss = loss_x * self.lambda_xy + loss_y * self.lambda_xy + \
                loss_w * self.lambda_wh + loss_h * self.lambda_wh + \
                loss_conf * self.lambda_conf + loss_cls * self.lambda_cls

            self.conf_mask[...] = 0
            self.noobj_mask[...] = 1
            self.tx[...] = 0
            self.ty[...] = 0
            self.tw[...] = 0
            self.th[...] = 0
            self.tconf[...] = 0
            self.tcls[...] = 0
            return loss, loss_x.item()* self.lambda_xy, loss_y.item()* self.lambda_xy, loss_w.item()* self.lambda_wh,\
                loss_h.item()* self.lambda_wh, loss_conf.item()* self.lambda_conf, loss_cls.item()* self.lambda_cls,recall
        else:

            # Add offset and scale with anchors

            # Results
            _scale = torch.Tensor([self.stride, self.stride] * 2).type(self.FloatTensor)
            output = torch.cat((pred_boxes.view(bs, -1, 4) * _scale,
                                conf.view(bs, -1, 1), pred_cls.view(bs, -1, self.num_classes)), -1)
            return output.data

    def get_target(self,pred_boxes, target, anchors, in_w, in_h, ignore_threshold):
        bs = target.size(0)
        # tx = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        # ty = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        # tw = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        # th = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        # tconf = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        # tcls = torch.zeros(bs, self.num_anchors, in_h, in_w, self.num_classes, requires_grad=False)
        nGT = 0
        nCorrect = 0
        for b in range(bs):
            for t in range(target.shape[1]):
                if target[b, t].sum() == 0:
                    continue
                nGT += 1
                # Convert to position relative to box
                gx = target[b, t, 1] * in_w
                gy = target[b, t, 2] * in_h
                gw = target[b, t, 3] * in_w
                gh = target[b, t, 4] * in_h
                # Get grid box indices
                gi = int(gx)
                gj = int(gy)
                # Get shape of gt box
                gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)

                # Get shape of anchor box
                anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((self.num_anchors, 2)),
                                                                  np.array(anchors)), 1))
                # Calculate iou between gt and anchor shapes
                anch_ious = bbox_iou(gt_box, anchor_shapes)
                # Where the overlap is larger than threshold set mask to zero (ignore)
                self.noobj_mask[b, anch_ious > ignore_threshold, gj, gi] = 0
                # Find the best matching anchor box
                best_n = np.argmax(anch_ious)

                if gi >= pred_boxes.shape[3]:
                    print(pred_boxes.shape, b, best_n, gj, gi)
                    gi = pred_boxes.shape[3] - 1

                if gj >= pred_boxes.shape[2]:
                    print(pred_boxes.shape, b, best_n, gj, gi)
                    gj = pred_boxes.shape[2] - 1

                gt_box = torch.FloatTensor(np.array([gx, gy, gw, gh])).unsqueeze(0)
                pred_box = pred_boxes[b, best_n, gj, gi].unsqueeze(0)
                # Masks
                self.conf_mask[b, best_n, gj, gi] = 1
                # Coordinates
                self.tx[b, best_n, gj, gi] = gx - gi
                self.ty[b, best_n, gj, gi] = gy - gj
                # Width and height
                self.tw[b, best_n, gj, gi] = math.log(gw/anchors[best_n][0] + 1e-16)
                self.th[b, best_n, gj, gi] = math.log(gh/anchors[best_n][1] + 1e-16)
                # object
                self.tconf[b, best_n, gj, gi] = 1
                # One-hot encoding of label
                self.tcls[b, best_n, gj, gi, int(target[b, t, 0])] = 1

                iou = bbox_iou(gt_box, pred_box,x1y1x2y2=False)
                if iou > 0.8:
                    nCorrect = nCorrect + 1
        return  nGT, nCorrect
