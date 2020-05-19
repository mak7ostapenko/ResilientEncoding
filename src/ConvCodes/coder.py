import numpy as np


class Coder:
    def __init__(self, num_inputs, num_outputs, code_len, summator_polynoms):
        # check the arguments
        assert num_outputs == summator_polynoms.shape[0], \
            "Number of polynoms have to be equal to number of outputs."
        assert code_len == summator_polynoms.shape[1], \
            "Number of polynom coefficients have to be equal to the length of code."
        uniq_elements = np.unique(summator_polynoms) == [0, 1]
        assert uniq_elements.all() == True, \
            "Polynom has to have only binary values."

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.code_len = code_len

        self.__reset_register_state()
        self.summator_polynoms = summator_polynoms

    def __reset_register_state(self):
        self.__register_state = np.zeros(self.code_len)

    def __update_register_state(self, symbol):
        """

        Args:
            symbol (int): input symbol into register.

        Returns:

        """
        self.__register_state = np.delete(self.__register_state, -1)
        self.__register_state = np.insert(self.__register_state, 0, symbol)

    def get_register_state(self):
        return self.__register_state

    def __set_register_state(self, state):
        assert np.array(state.shape == self.__register_state.shape).all() == True, \
            "Size of states has to be the same."

        self.__register_state = state

    def __get_coder_response(self):
        coder_response = list()
        for polynom in self.summator_polynoms:
            sum = (polynom * self.get_register_state()).sum()

            if sum % 2 == 0:
                coder_response.append(0)
            else:
                coder_response.append(1)

        return coder_response

    def encode(self, input_sequence, reset_register_state=True, register_state=None):
        """

        Args:
            input_sequence (np.array): with shape=(n).
            reset_register_state (bool): if True then make all register states equal to zero.

        Returns:
            coder_response (np.array): coder response on an input sequence.
        """
        assert len(input_sequence.shape) == 1, \
            "Input sequence has to be one dimensional."
        assert (np.unique(input_sequence).any() <= True).all() == True, \
            "Input sequence has to have only binary values."

        if reset_register_state and register_state is not None:
            self.__set_register_state(register_state)

        elif reset_register_state:
            self.__reset_register_state()

        sumators_responses = np.zeros((input_sequence.shape[0], self.num_outputs))
        for ind, symbol in enumerate(input_sequence[::-1]):
            self.__update_register_state(symbol)
            sumators_responses[ind, :] = self.__get_coder_response()

        coder_response = sumators_responses.reshape(-1)

        return coder_response


def encoding_test():
    num_inputs = 1
    num_outputs = 2
    code_len = 4
    summator_polynoms = np.array([
        [1, 1, 1, 1],
        [1, 0, 0, 1],
    ])

    coder = Coder(num_inputs, num_outputs, code_len, summator_polynoms)

    input_sequence = np.array([0, 0, 0, 1])
    response = coder.encode(input_sequence)

    assert np.array(response == [1, 1, 1, 0, 1, 0, 1, 1]).all() == True

    print('inp len = ', len(input_sequence))
    print('resp len = ', len(response))
    print('res = ', response)


if __name__ == '__main__':
    encoding_test()
