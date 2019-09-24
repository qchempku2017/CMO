import numpy as np
import os

#from pyabinitio.compressive.src.bregman_src import bregman_functions


def split_bregman(A, f, MaxIt=1e5, tol=1e-5, mu=10, l=10, quiet=False):
    """
    ! Performs split Bregman iterations using conjugate gradients (CG) for the
    ! L2 minimizations and shrinkage for L1 minimizations.
    !
    !    u = arg min_u { ||u||_1 + mu/2 ||Au-f||_2^2 }
    !
    ! The algorithm is described in T. Goldstein and S. Osher,
    ! "The split Bregman method for L1 regularized problems",
    ! SIAM Journal on Imaging Sciences Volume 2 Issue 2, Pages 323-343 (2009).
    !
    ! Input parameters:
    !   MaxIt   - Number of outer split Bregman loops
    !   tol     - Required tolerance for the residual. The algorithm stops when
    !             tol > ||Au-f||_2 / ||f||_2
    !   mu      - weight for the L1 norm of the solution
    !   l       - weight for the split constraint (affects speed of convergence, not the result)
    !   N       - number of unknown expansion coefficients [ =length(u) ]
    !   M       - number of measurements [ =length(f) ]
    !   A       - sensing matrix A
    !   f       - array with measured signal values (of length M)
    !   u       - solution array (of length N)
    !
    ! Output:
    !   u       - solution
    integer, intent(in)             :: MaxIt, N, M
    double precision, intent(in)    :: tol, mu, lambda
    double precision,intent(in)     :: A(M,N), f(M)
    double precision,intent(inout)  :: u(N)

    integer k, MaxCGit
    double precision crit1, crit2
    double precision, allocatable:: uprev(:), b(:), d(:), bp(:)
    """
    f = np.array(f, dtype=np.double, order='F')
    A = np.array(A, order='F')
    u = np.ones(A.shape[1], dtype=np.double, order='F')

    if quiet:
        null_fds = [os.open(os.devnull, os.O_RDWR) for x in xrange(2)]
        # save the current file descriptors to a tuple
        save = os.dup(1), os.dup(2)
        # put /dev/null fds on 1 and 2
        os.dup2(null_fds[0], 1)
        os.dup2(null_fds[1], 2)

    bregman_functions.splitbregman(MaxIt, tol, mu, l,  a=A, f=f, u=u)

    if quiet:
        # restore file descriptors
        os.dup2(save[0], 1)
        os.dup2(save[1], 2)
        # close the temporary fds
        os.close(null_fds[0])
        os.close(null_fds[1])
        os.close(save[0])
        os.close(save[1])

    return u
