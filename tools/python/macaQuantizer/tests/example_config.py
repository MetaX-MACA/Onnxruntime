argmax_name_list = [
        "test_argmax_default_axis_example",
        "test_argmax_default_axis_example_select_last_index", 
        "test_argmax_default_axis_random",
        "test_argmax_default_axis_random_select_last_index",
        "test_argmax_keepdims_example",
        "test_argmax_keepdims_example_select_last_index",
        "test_argmax_keepdims_random",
        "test_argmax_keepdims_random_select_last_index",
        "test_argmax_negative_axis_keepdims_example",
        "test_argmax_negative_axis_keepdims_example_select_last_index",
        "test_argmax_negative_axis_keepdims_random",
        "test_argmax_negative_axis_keepdims_random_select_last_index",
        "test_argmax_no_keepdims_example",
        "test_argmax_no_keepdims_example_select_last_index",
        "test_argmax_no_keepdims_random", 
        "test_argmax_no_keepdims_random_select_last_index"
]

argmin_name_list = [
        "test_argmin_default_axis_example",
        "test_argmin_default_axis_example_select_last_index",
        "test_argmin_default_axis_random",
        "test_argmin_default_axis_random_select_last_index",
        "test_argmin_keepdims_example",
        "test_argmin_keepdims_example_select_last_index",
        "test_argmin_keepdims_random",
        "test_argmin_keepdims_random_select_last_index",
        "test_argmin_negative_axis_keepdims_example",
        "test_argmin_negative_axis_keepdims_example_select_last_index",
        "test_argmin_negative_axis_keepdims_random",
        "test_argmin_negative_axis_keepdims_random_select_last_index",
        "test_argmin_no_keepdims_example",
        "test_argmin_no_keepdims_example_select_last_index",
        "test_argmin_no_keepdims_random",
        "test_argmin_no_keepdims_random_select_last_index"

]

averagepool_name_list = [
    "test_averagepool_1d_default",
    "test_averagepool_2d_ceil",
    "test_averagepool_2d_default",
    # # "test_averagepool_2d_pads",                     # pad not supported
    # # "test_averagepool_2d_pads_count_include_pad",   # pad not supported
    "test_averagepool_2d_precomputed_pads",
    "test_averagepool_2d_precomputed_pads_count_include_pad",
    "test_averagepool_2d_precomputed_same_upper",
    "test_averagepool_2d_precomputed_strides",
    "test_averagepool_2d_same_lower",      # X
    "test_averagepool_2d_same_upper",      # X
    "test_averagepool_2d_strides",         # X
    "test_averagepool_3d_default"          # X
]

ceil_name_list = ["test_ceil", "test_ceil_example"]

depthtospace_name_list = [
    "test_depthtospace_crd_mode",
    "test_depthtospace_crd_mode_example",
    "test_depthtospace_dcr_mode",
    "test_depthtospace_example"
]


cumsum_name_list = [
    "test_cumsum_1d",
    "test_cumsum_1d_exclusive",
    "test_cumsum_1d_reverse",
    "test_cumsum_1d_reverse_exclusive",
    "test_cumsum_2d_axis_0",
    "test_cumsum_2d_axis_1",
    "test_cumsum_2d_negative_axis"
]

einsum_name_list = [
    "test_einsum_batch_diagonal",
    "test_einsum_batch_matmul",
    "test_einsum_inner_prod",
    "test_einsum_sum",
    "test_einsum_transpose"
]



maxunpool_tst_list = [
    "test_maxunpool_export_with_output_shape",
    "test_maxunpool_export_without_output_shape"
]


reducemin_tst_list = [
    "test_reduce_min_default_axes_keepdims_example",
    "test_reduce_min_default_axes_keepdims_random",
    "test_reduce_min_do_not_keepdims_example",
    "test_reduce_min_do_not_keepdims_random",
    "test_reduce_min_keepdims_example",
    "test_reduce_min_keepdims_random",
    "test_reduce_min_negative_axes_keepdims_example",
    "test_reduce_min_negative_axes_keepdims_random"
]


reduceprod_tst_list = [
    "test_reduce_prod_default_axes_keepdims_example",
    "test_reduce_prod_default_axes_keepdims_random",
    "test_reduce_prod_do_not_keepdims_example",
    "test_reduce_prod_do_not_keepdims_random",
    "test_reduce_prod_keepdims_example",
    "test_reduce_prod_keepdims_random",
    "test_reduce_prod_negative_axes_keepdims_example",
    "test_reduce_prod_negative_axes_keepdims_random",
]


scatter_elements_tst_list = [
    "test_scatter_elements_with_axis",
    "test_scatter_elements_with_duplicate_indices",
    "test_scatter_elements_with_negative_indices",
    "test_scatter_elements_without_axis",
]


split2squence_tst_list=[
    "test_split_to_sequence_1",
    "test_split_to_sequence_2",
    "test_split_to_sequence_nokeepdims"
]



xor_tst_list = [
    "test_xor2d",
    "test_xor3d",
    "test_xor4d",
    "test_xor_bcast3v1d",
    "test_xor_bcast3v2d",
    "test_xor_bcast4v2d",
    "test_xor_bcast4v3d",
    "test_xor_bcast4v4d"
]



