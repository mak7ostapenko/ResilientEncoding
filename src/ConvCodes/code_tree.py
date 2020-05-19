import numpy as np
from binarytree import tree

from src.ConvCodes.coder import Coder


def format_encoded_value_for_tree(code):
    code[code==0] = 8
    str_code = [str(int(value)) for value in code]
    out_value = int(''.join(str_code))

    return out_value


def build_code_tree(coder, tree_height=-1):
    """
    Notes:
        On the diagram 8 is equal to 0 because of
        tree.value has to be int number.

    Args:
        coder:
        tree_height:

    Returns:

    """
    if tree_height == -1:
        tree_height = coder.code_len

    code_tree = tree(height=tree_height, is_perfect=True)
    # set value for root
    code_tree.value = 18

    state_dict = dict()
    state_dict[0] = coder.get_register_state()

    for ind, node in enumerate(code_tree.levelorder):
        left_son_ind = 2 * ind + 1
        right_son_ind = 2 * ind + 2

        if left_son_ind < len(code_tree.levelorder):
            # encode for the left son
            node.left.value = format_encoded_value_for_tree(
                coder.encode(
                    input_sequence=np.array([1]),
                    reset_register_state=True,
                    register_state=state_dict[ind])
            )
            state_dict[left_son_ind] = coder.get_register_state()

            # encode for the right son
            node.right.value = format_encoded_value_for_tree(
                coder.encode(
                    input_sequence=np.array([0]),
                    reset_register_state=True,
                    register_state=state_dict[ind])
            )
            state_dict[right_son_ind] = coder.get_register_state()

    return code_tree


def test():
    num_inputs = 1
    num_outputs = 3
    code_len = 3
    summator_polynoms = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 0, 1]
    ])

    coder = Coder(num_inputs, num_outputs, code_len, summator_polynoms)
    code_tree = build_code_tree(coder)
    print(code_tree)


if __name__ == '__main__':
    test()