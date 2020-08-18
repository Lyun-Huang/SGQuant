#!/usr/bin/env python3
from collections import namedtuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import InplaceFunction, Function
from collections import defaultdict
import pickle
import os
import sys
import struct
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from scipy import stats
#import math
import bisect

# sns.set(color_codes=True)

QParams = namedtuple('QParams', ['range', 'zero_point', 'num_bits'])

_DEFAULT_FLATTEN = (1, -1)
_DEFAULT_FLATTEN_GRAD = (0, -1)

#将x的维度变成和x_full相同
def _deflatten_as(x, x_full):
    shape = list(x.shape) + [1] * (x_full.dim() - x.dim())
    return x.view(*shape)


def calculate_qparams(x, num_bits, flatten_dims=_DEFAULT_FLATTEN, reduce_dim=0, reduce_type='mean', keepdim=False, true_zero=False):
    with torch.no_grad():
        x_flat = x.flatten(*flatten_dims)
        if x_flat.dim() == 1:
            min_values = _deflatten_as(x_flat.min(), x)
            max_values = _deflatten_as(x_flat.max(), x)
        else:
            min_values = _deflatten_as(x_flat.min(-1)[0], x)
            max_values = _deflatten_as(x_flat.max(-1)[0], x)
        if reduce_dim is not None:
            if reduce_type == 'mean':
                min_values = min_values.mean(reduce_dim, keepdim=keepdim)
                max_values = max_values.mean(reduce_dim, keepdim=keepdim)
            else:
                min_values = min_values.min(reduce_dim, keepdim=keepdim)[0]
                max_values = max_values.max(reduce_dim, keepdim=keepdim)[0]

        # TODO: re-add true zero computation
        range_values = max_values - min_values
        return QParams(range=range_values, zero_point=min_values, num_bits=num_bits)


class UniformQuantize(InplaceFunction):

    @staticmethod
    def forward(ctx, input, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN,
                reduce_dim=0, dequantize=True, signed=False, stochastic=False, inplace=False):

        ctx.inplace = inplace

        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        if qparams is None:
            assert num_bits is not None, "either provide qparams of num_bits to quantize"
            qparams = calculate_qparams(
                input, num_bits=num_bits, flatten_dims=flatten_dims, reduce_dim=reduce_dim)

        # print(output)
        # zero_point = torch.min(output) 
        # range_val = torch.max(output) - torch.min(output) 
        zero_point = qparams.zero_point
        range_val = qparams.range
        # print(qparams.zero_point)
        # print(torch.min(output))
        num_bits = qparams.num_bits
        qmin = -(2.**(num_bits - 1)) if signed else 0.
        qmax = qmin + 2.**num_bits - 1.
        # scale = qparams.range / (qmax - qmin)
        scale = range_val / (qmax - qmin)
        with torch.no_grad():
            output.add_(qmin * scale - zero_point).div_(scale)
            if stochastic:
                noise = output.new(output.shape).uniform_(-0.5, 0.5)
                output.add_(noise)

            # quantize
            output.clamp_(qmin, qmax).round_()

            if dequantize:
                output.mul_(scale).add_(
                    zero_point - qmin * scale)  # dequantize
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # straight-through estimator
        grad_input = grad_output
        return grad_input, None, None, None, None, None, None, None, None


def quantize(x, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN, reduce_dim=0, 
    dequantize=True, signed=False, stochastic=False, inplace=False):
    return UniformQuantize().apply(
        x, num_bits, qparams, flatten_dims, reduce_dim, dequantize, signed, stochastic, inplace)

def float_to_bin(num):
    return format(struct.unpack('!I', struct.pack('!f', num))[0], '032b')

def bin_to_float(binary):
    return struct.unpack('!f',struct.pack('!I', int(binary, 2)))[0]

#def float2tf32(f, num_e_bits=8, num_m_bits=23, bias=127., dtype=torch.float32):
#    if torch.is_tensor(f) is not True:
#        f=torch.tensor([f])
#    else:
#        f=f.item()
#        f=torch.tensor([f])
#    binary=float2bit(f, num_e_bits=num_e_bits, num_m_bits=num_m_bits, bias=bias)
##    print(binary)
#    for _ in range(19,32):
#        binary[0, _]=0
##    print(binary)
#    back=bit2float(binary).item()
#    return back

def float2tf32(f):
    binary=float_to_bin(f)
#    print(binary)
    binary=list(binary)
    for i in range(19,32):
        binary[i]='0'
    binary=''.join(binary)
#    print(binary)
    return bin_to_float(binary)

def quant_tf32(x):
    q=torch.randn_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i,j].item() == 0:
                q[i,j]=torch.tensor(0.)
            else:
                q[i,j]=torch.tensor(float2tf32(x[i,j].item()))
    return q

def float2bf16(f):
    binary=float_to_bin(f)
#    print(binary)
    binary=list(binary)
    for i in range(16,32):
        binary[i]='0'
    binary=''.join(binary)
#    print(binary)
    return bin_to_float(binary)

def quant_bf16(x):
    q=torch.randn_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i,j].item() == 0:
                q[i,j]=torch.tensor(0.)
            else:
                q[i,j]=torch.tensor(float2bf16(x[i,j].item()))
    return q

def main():
    x=3.224153
    print(float2tf32(x))
    pass
    # torch.manual_seed(4)
    # x = torch.randn((8, 8), requires_grad=True)
    # w1 = torch.randn((8, 8), requires_grad=True)
    # z1 = F.linear(x, w1)
    # a1 = F.relu(z1)
    # print(a1)
    # print(id(a1))
    # print(q_a1)

    # w2 = torch.randn((8, 8), requires_grad=True)
    # z2 = F.linear(a1, w2)
    # a2 = F.relu(z2)

    # a1_q = quantize(a1, num_bits=8, dequantize=True)
    # w2_q = quantize(w2, num_bits=8, dequantize=True)
    # z2_q = F.linear(a1_q, w2_q)
    # a2_q = F.relu(z2_q)
    
    # diff = (a2_q - a1_q)
    # loss = torch.sqrt(diff * diff)
    # print(loss)
    # a1 = a1.to(torch.uint8)
    # print(a1)
    # print(torch.randint(100, (4,4), dtype=torch.uint8))
    # print(id(a1))
    # a1.data.zero_()

    # loss = a2.mean()
    # print(loss)
    # loss.backward()

    # print(w2.grad)

    # q_x = quantize(x, num_bits=4, dequantize=True)
    # print(q_x)

    # q_x = quantize(x, num_bits=4, dequantize=False)
    # print(q_x)

