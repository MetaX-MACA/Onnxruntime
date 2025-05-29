import torch


# implement ArgMax, ArgMin to support 'select_last_index'
def last_index_impl(op, input, axis, keep_dim =True):
    descend = True if op.type in {"ArgMin"} else False
    sotr_index = torch.argsort(input, dim=axis, descending=descend)
    last_index = torch.select(sotr_index, dim=axis, index=-1)
    output = last_index
    if keep_dim:
        axis_ = axis + input.ndim if axis < 0 else axis
        keep_shape = [1 if i==axis_ else input.shape[i] for i in range(input.ndim)]
        output = last_index.reshape(keep_shape)

    return output


# implement CumSum to support 'exclusive'
def _arbitrary_dim_shift_and_insert_zero(
    input_tensor: torch.Tensor,
    insert_dim: int,
) -> torch.Tensor:
    # single item shift
    slice_index, insertion = [[slice(None)] * len(input_tensor.shape)] * 2
    insert_dim_size = input_tensor.shape[insert_dim]

    slice_index[insert_dim] = slice(0, -1)
    slice_index = tuple(slice_index)
    tensor_slice = input_tensor[slice_index]

    insert_index = torch.arange(start=1, end=insert_dim_size, dtype=torch.int64, device=input_tensor.device)
    index_shape = [1] * len(input_tensor.shape)
    index_shape[insert_dim] = insert_dim_size - 1

    insert_index = torch.reshape(insert_index, index_shape)
    insert_index = insert_index + torch.zeros_like(tensor_slice, dtype=torch.int64, device=input_tensor.device)

    input_tensor = torch.scatter(
        input=input_tensor,
        dim=insert_dim,
        index=insert_index,
        src=tensor_slice,
    )

    insertion[insert_dim] = slice(0, 1)
    insertion = tuple(insertion)
    input_tensor[insertion] = 0

    return input_tensor