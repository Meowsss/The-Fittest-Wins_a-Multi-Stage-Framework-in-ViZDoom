# -*- coding: utf-8 -*-
import tensorflow as tf
import horovod.tensorflow as hvd
import numpy as np
from tleague.utils import logger


# PiecewiseFusion
def pw_fusion(pw_fusion_schedule: str, grads_and_vars: list, batch_size: int, sparse_as_dense: bool = True):
  logger.log("Using PieceWise Fusion with schedule {}.".format(pw_fusion_schedule))
  grads_for_pf = [grad for grad, var in grads_and_vars]
  vars_for_pf = [var for grad, var in grads_and_vars]

  pw = PiecewiseFusion(sparse_as_dense=sparse_as_dense)
  pieces = pw_fusion_schedule.split(';')
  boundaries = []
  for piece in pieces:
    boundaries.append(int(piece))
  if (len(boundaries) == 1) and (boundaries[0] <= 0):
    grads_and_vars = pw.non_fusion(grads_for_pf, vars_for_pf)
  elif (len(boundaries) >= 1) and (boundaries[0] > 0):
    grads_and_vars, _ = pw.piecewise_fusion(grads=grads_for_pf,
                                            params=vars_for_pf,
                                            boundaries=boundaries,
                                            loss_scale=None,
                                            num_gpus=hvd.size(),
                                            batch_size=batch_size,
                                            network_name='policy')
  else:
    raise ValueError('boundaries parameter is invalid')
  return grads_and_vars


class PiecewiseFusion(object):
    def __init__(self, sparse_as_dense=True):
        self._sparse_as_dense = sparse_as_dense

    def _sparse_to_dense(self, grads):
        grads = [tf.convert_to_tensor(grad)
                 if grad is not None and isinstance(grad, tf.IndexedSlices)
                 else grad for grad in grads]
        return grads

    def non_fusion(self, grads, params):
        if self._sparse_as_dense:
            grads = self._sparse_to_dense(grads)
        g_grads = []
        for grad in grads:
          if grad == None:
            g_grads.append(grad)
          else:
            g_grads.append(hvd.allreduce(grad, average=True, device_dense=''))
        gradvars = list(zip(g_grads, params))
        return gradvars

    def piecewise_fusion(self, grads, params, boundaries, loss_scale, num_gpus, batch_size, network_name):
        if self._sparse_as_dense:
            grads = self._sparse_to_dense(grads)
        gradvars = list(zip(grads, params))
        # check suitable boundaries
        for d, (grad, param) in enumerate(gradvars):
            if grad !=None and hasattr(grad, "shape"):
              print(network_name, d, grad.shape.num_elements()) 
        # check suitable boundaries
        tensors_with_shapes = []
        allgrads = []
        grads_beforeavg = []
        grads_afteravg = []
        params_new = []
        flat_tensors ={}
        orig_shapes = {}
        orig_sizes = {}
        g_grads = []
        #pieces = piecewise_fusion_schedule.split(';')
        #for piece in pieces:
        #    boundaries.append(int(piece))
        print("boundaries", boundaries)
        indexs = np.arange(len(boundaries)+1)
        for i in range(len(boundaries)+1):
              flat_tensors[i] =[]
              orig_shapes[i] = []
              orig_sizes[i] = []

        for d, (grad, param) in enumerate(gradvars):
            if grad !=None:
              if not (hasattr(grad, "shape")): 
                allgrads.append(hvd.allreduce(grad, average=True, device_dense=''))                  
                params_new.append(param)
              else:
                grad1=(grad if loss_scale is None
                           else grad * (1. / loss_scale))
                grads_beforeavg.append(grad1)
                params_new.append(param)

                if d < boundaries[0]:
                  index = indexs[0]
                  flat_tensors[index].append(tf.reshape(grad1, [-1]))
                  orig_shapes[index].append(grad1.shape)
                  orig_sizes[index].append(grad1.shape.num_elements())
                if d >= boundaries[-1]:
                  index = indexs[-1]
                  flat_tensors[index].append(tf.reshape(grad1, [-1]))
                  orig_shapes[index].append(grad1.shape)
                  orig_sizes[index].append(grad1.shape.num_elements())

                for low, high, index in zip(boundaries[:-1], boundaries[1:], indexs[1:-1]):
                  if (d >= low) and (d < high):
                    flat_tensors[index].append(tf.reshape(grad1, [-1]))
                    orig_shapes[index].append(grad1.shape)
                    orig_sizes[index].append(grad1.shape.num_elements())

        for i in range(len(boundaries)+1):
            if(hvd.rank() == 0):
                print("orig_sizes: ", i, orig_sizes[i])

            if len(flat_tensors[i]) == 0:
              tensors_with_shapes.append([])
            elif len(flat_tensors[i]) == 1:
              concatenated_grad = flat_tensors[i][0]
              concatenated_grad_hvd = hvd.allreduce(concatenated_grad, average=True, device_dense='')
              tensors_with_shapes.append([tf.reshape(concatenated_grad_hvd, orig_shapes[i][0])])
            else:
              concatenated_grad = tf.concat(flat_tensors[i], 0)
              concatenated_grad_hvd = hvd.allreduce(concatenated_grad, average=True, device_dense='')
              #concatenated_grad_hvd = tf.cast(concatenated_grad_hvd, tf.float32)
              tensors_with_sizes = tf.split(concatenated_grad_hvd, orig_sizes[i])
              tensors_with_shapes.append([tf.reshape(grad, shape)
                                          for grad, shape in zip(tensors_with_sizes, orig_shapes[i])])

        for i in range(len(boundaries)+1):
            allgrads.extend(tensors_with_shapes[i])

        grads_afteravg = allgrads
        g_grad_norm_beforeavg = tf.global_norm(grads_beforeavg)
        g_grad_norm_beforeavg = g_grad_norm_beforeavg*g_grad_norm_beforeavg
        g_grad_norm_afteravg = tf.global_norm(grads_afteravg)
        g_grad_norm_afteravg = g_grad_norm_afteravg*g_grad_norm_afteravg

        if num_gpus == 1:
            batchsize=float(batch_size)
            noise_g = batchsize*g_grad_norm_afteravg-batchsize*g_grad_norm_beforeavg+1
            noise_s = g_grad_norm_beforeavg-g_grad_norm_afteravg+1
        else:
            large_batchsize=float(num_gpus*batch_size)
            small_batchsize=float(batch_size)
            alpha = 1/(large_batchsize-small_batchsize)
            beta = 1/(1/small_batchsize-1/large_batchsize)
            noise_g = alpha*(large_batchsize*g_grad_norm_afteravg-small_batchsize*g_grad_norm_beforeavg)
            noise_s = beta*(g_grad_norm_beforeavg-g_grad_norm_afteravg)
        noise_scale=tf.div(noise_s,noise_g)
        relnoise_scale = noise_scale
        max_noisescale = hvd.allreduce(relnoise_scale, average=True, device_dense='')

        gradvars = list(zip(allgrads, params_new))
        #for d, (grad, param) in enumerate(gradvars):
        #  print("after grad: ", d, grad, param)
        return gradvars, max_noisescale
