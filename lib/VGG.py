def VGG16N(x, n_classes, IS_PRETRAIN):
    import tools

    x = tools.conv('conv1_1', x, 64, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=IS_PRETRAIN[0])
    x = tools.conv('conv1_2', x, 64, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=IS_PRETRAIN[1])
    x = tools.pool('pool1', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)

    x = tools.conv('conv2_1', x, 128, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=IS_PRETRAIN[2])
    x = tools.conv('conv2_2', x, 128, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=IS_PRETRAIN[3])
    x = tools.pool('pool2', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)

    x = tools.conv('conv3_1', x, 256, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=IS_PRETRAIN[4])
    x = tools.conv('conv3_2', x, 256, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=IS_PRETRAIN[5])
    x = tools.conv('conv3_3', x, 256, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=IS_PRETRAIN[6])
    x = tools.pool('pool3', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)

    x = tools.conv('conv4_1', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=IS_PRETRAIN[7])
    x = tools.conv('conv4_2', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=IS_PRETRAIN[8])
    x = tools.conv('conv4_3', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=IS_PRETRAIN[9])
    x = tools.pool('pool3', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)

    x = tools.conv('conv5_1', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=IS_PRETRAIN[10])
    x = tools.conv('conv5_2', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=IS_PRETRAIN[11])
    x = tools.conv('conv5_3', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=IS_PRETRAIN[12])
    x = tools.pool('pool3', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)

    x = tools.FC_layer('fc6', x, out_nodes=4096,is_pretrain=IS_PRETRAIN[13])
    x = tools.batch_norm(x)
    x = tools.FC_layer('fc7', x, out_nodes=4096,is_pretrain=IS_PRETRAIN[14])
    x = tools.batch_norm(x)
    x = tools.final_layer('fc8', x, out_nodes=n_classes)
    return x
