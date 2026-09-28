def create_param_groups(model):
    param_groups = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_groups.append({'params': param, 'lr': None})
    return param_groups

def assign_lr(param_groups,
              base_lr,
              layer_list,
              decay_factor):

    '''Assign LR to param group based on layer, the bigger the layer the bigger the decay factor (larger LR)'''

    n = layer_list[-1]

    for i, param_group in enumerate(param_groups):
        param_group['lr'] = base_lr * (decay_factor ** (n - layer_list[i]))

def identify_layer_number(model):
    layer_num     = 0
    layer_list    = []
    previous_lyr  = None
    current_lyr   = None
    for name, param in model.named_parameters():
      if param.requires_grad:
        if previous_lyr is None:
            previous_lyr  = name.split('.')
            layer_list.append(layer_num)
            continue

        current_lyr   = name.split('.')

        if current_lyr[0] != previous_lyr[0]:
            layer_num     = layer_num+1
            layer_list.append(layer_num)
            previous_lyr  = current_lyr
            continue

        if current_lyr[1] == 'bias':
            layer_list.append(layer_num)
            previous_lyr  = current_lyr
            continue

        if current_lyr[1] != previous_lyr[1]:
            layer_num     = layer_num+1
            layer_list.append(layer_num)
            previous_lyr  = current_lyr
            continue

        if current_lyr[2] == 'bias':
            layer_list.append(layer_num)
            previous_lyr  = current_lyr
            continue

        if current_lyr[2] != previous_lyr[2]:
            layer_num     = layer_num+1
            layer_list.append(layer_num)
            previous_lyr  = current_lyr
        else:
            layer_list.append(layer_num)
            previous_lyr  = current_lyr

    return layer_list