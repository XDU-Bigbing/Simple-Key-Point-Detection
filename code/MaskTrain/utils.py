from __future__ import print_function

from collections import defaultdict, deque
import datetime
import pickle
import time

import torch
import torch.distributed as dist

import errno
import os


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        # 队列的用途不懂
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    # 更新队列
    def update(self, value, n=1):
        self.deque.append(value)
        # count 队列的元素
        self.count += n
        # 队列元素中的数值之和
        self.total += value * n

    # 同步进程信息
    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        # 不支持并行直接返回
        if not is_dist_avail_and_initialized():
            return
        # 支持并行的情况下
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        # 用于同步进程，阻塞等待所有的进程进入
        dist.barrier()
        # AllReduce通信方法获取多卡上的平均梯度，操作后每个卡上的每一个bit，t都相同
        # t 是广播通信的输入和输出，就地操作
        dist.all_reduce(t)
        # 转为列表
        t = t.tolist()
        # [self.count, self.total]的 Tensor 转为列表，在取出前两个元素
        self.count = int(t[0])
        self.total = t[1]

    # 装饰器，方法当属性用
    @property
    # 返回中位数
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    # tensor 的平均值
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    # 不懂
    def global_avg(self):
        return self.total / self.count

    @property
    # 队列的最大值 value
    def max(self):
        return max(self.deque)

    @property
    # 取出刚压入队列的元素
    def value(self):
        return self.deque[-1]

    # 键不存在时显示的东西
    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): 所有的误差
        average (bool): 对所有卡上误差求和后平均，之后所有进程会有平均的结果，将结果返回
    """
    # 进程数量
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    # 此时不梯度下降
    with torch.no_grad():
        names = []
        values = []
        # 对分类损失，回归损失，mask 损失进行排序
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        # 将损失拼接
        values = torch.stack(values, dim=0)
        # 阻塞等待通信完成
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        # defaultdict 字典
        # 当键不存在时，打印 SmoothedValue 中提供的字符串
        # 字典的值是 SmoothedValue 类型
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        # 获取传入的参数
        for k, v in kwargs.items():
            # 获取数值
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            # SmoothedValue 类中
            # 更新每一个键的队列？
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        # self.__dict__ 函数额外添加的属性 或
        # 类的属性与方法
        # https://stackoverflow.com/questions/19907442/explain-dict-attribute
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        # key value 打印 loss
        return self.delimiter.join(loss_str)

    # 同步字典所有的值
    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    # 字典添加元素
    def add_meter(self, name, meter):
        self.meters[name] = meter

    # 返回训练数据
    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        # 获取全部数据的长度
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = self.delimiter.join([
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}',
            'max mem: {memory:.0f}'
        ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            # 返回一个调用对象
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(log_msg.format(
                    i, len(iterable), eta=eta_string,
                    meters=str(self),
                    time=str(iter_time), data=str(data_time),
                    memory=torch.cuda.max_memory_allocated() / MB))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def collate_fn(batch):
    return tuple(zip(*batch))


# 计算新的学习率
def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

    # 初始学习率乘上这个函数的返回值作为新的学习率
    def f(x):
        # 传入的参数是 epoch ，大于 1000 不在衰减
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        # 1000 - x + x / 1000
        return warmup_factor * (1 - alpha) + alpha

    # 先执行 return 
    # 对每一个参数设置初始化的学习率衰减次数通过函数 f
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


# 判断并行库是否支持，如果不支持，那么暴露任何接口
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return False


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)
