class GraphBlasAlgorithmException(Exception):
    pass


class ConvergenceFailure(GraphBlasAlgorithmException):
    pass


class EmptyGraphError(GraphBlasAlgorithmException):
    pass


class PointlessConcept(GraphBlasAlgorithmException):
    pass
