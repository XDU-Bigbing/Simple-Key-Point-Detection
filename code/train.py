import utils
from data import Loader


if __name__ == "__main__":

    JSONPATH = '../data/tile_round1_train_20201231/train_annos.json'
    DATAPATH = "../data/tile_round1_train_20201231/train_imgs/"
    OUTFILE = 'output.txt'
    
    # 获取有多少数据
    data_num = utils.get_data_num(JSONPATH)
    
    # 最开始创建时覆盖原来文件
    with open(OUTFILE, 'w') as f:
        f.write("Total data number is {} \n".format(data_num))

    # 数据划分，60%训练，20%测试，20%验证
    train_scale, test_scale, valid_scale = 0.6, 0.2, 0.2
    sum_str = "The sum number of train, test, valid must equal to 1"
    assert train_scale + test_scale + valid_scale == 1.0, sum_str
    
    # 获取训练集、测试集合的数量
    train_num, test_num, valid_num = utils.get_train_test_valid_num(
        data_num=data_num, 
        train_scale=train_scale, 
        test_scale=test_scale, 
        valid_scale=valid_scale)
    
    # 9138, 3046, 3046
    # print(train_num, test_num, valid_num, sep=', ')
    # 之后写文件就追加
    with open(OUTFILE, 'a') as f:
        f.write("Train data number is {} \n".format(train_num))
        f.write("Test data number is {} \n".format(test_num))
        f.write("Valid data number is {} \n".format(valid_num))

    # 加载所有数据集
    DATA = Loader(sum_num=data_num, 
        data_path=DATAPATH,
        json_path=JSONPATH,)

    # 一批训练的数据量
    batchsz = 16