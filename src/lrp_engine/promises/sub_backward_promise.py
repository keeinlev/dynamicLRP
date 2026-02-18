from .add_backward_promise import AddBackwardPromise


class SubBackwardPromise(AddBackwardPromise):

    @property
    def op_result(self):
        """Returns the forward result of the operation when applied on the promise args"""
        return self.arg1 - self.arg2
