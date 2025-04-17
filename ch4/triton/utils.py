import os
from pdb import set_trace as breakpoint

def check_tensors_gpu_ready(*tensors):
    """
    Verify that tensors are properly prepared for GPU computation.
    
    This function checks that all tensors are contiguous in memory and on the CUDA device,
    which are requirements for efficient GPU computation with Triton.
    
    Args:
        *tensors: Variable number of PyTorch tensors to check
        
    Raises:
        AssertionError: If any tensor is not contiguous or not on CUDA
    """
    for t in tensors:
        assert t.is_contiguous, "A tensor is not contiguous"
        # Skip CUDA check if we're in interpretation mode (for debugging)
        if not os.environ.get('TRITON_INTERPRET') == '1': assert t.is_cuda, "A tensor is not on cuda"

def test_pid_conds(conds, pid_0=[0], pid_1=[0], pid_2=[0]):
    """
    Test if conditions on program IDs (pids) are fulfilled.
    
    This utility function helps with debugging Triton kernels by allowing
    conditional checks on program IDs (block indices).
    
    Args:
        conds (str): String of comma-separated conditions to check.
                     Each condition consists of an operator and a value.
        pid_0 (list): List containing the program ID for dimension 0
        pid_1 (list): List containing the program ID for dimension 1
        pid_2 (list): List containing the program ID for dimension 2
        
    Examples:
        '=0'     - checks that pid_0 == 0
        ',>1'    - checks that pid_1 > 1
        '>1,=0'  - checks that pid_0 > 1 and pid_1 == 0
        
    Returns:
        bool: True if all conditions are met, False otherwise
    
    Raises:
        ValueError: If an invalid operator is used in conditions
    """
    pids = pid_0[0], pid_1[0], pid_2[0]
    conds = conds.replace(' ','').split(',')
    for i, (cond, pid) in enumerate(zip(conds, pids)):
        if cond=='': continue
        op, threshold = cond[0], int(cond[1:])
        if op not in ['<','>','>=','<=','=', '!=']: raise ValueError(f"Rules may only use these ops: '<','>','>=','<=','=', '!='. Invalid rule: '{cond}'.")
        op = '==' if op == '=' else op
        if not eval(f'{pid} {op} {threshold}'): return False
    return True


def breakpoint_if(conds, pid_0=[0], pid_1=[0], pid_2=[0]):
    """
    Conditionally trigger a debugger breakpoint based on program ID conditions.
    
    This is useful for debugging specific blocks in a Triton kernel.
    
    Args:
        conds (str): Conditions string to test (see test_pid_conds)
        pid_0 (list): List containing the program ID for dimension 0
        pid_1 (list): List containing the program ID for dimension 1
        pid_2 (list): List containing the program ID for dimension 2
    """
    if test_pid_conds(conds, pid_0, pid_1, pid_2): breakpoint()

def print_if(txt, conds, pid_0=[0], pid_1=[0], pid_2=[0]):
    """
    Conditionally print a message based on program ID conditions.
    
    This is useful for debugging specific blocks in a Triton kernel.
    
    Args:
        txt (str): Text to print if conditions are met
        conds (str): Conditions string to test (see test_pid_conds)
        pid_0 (list): List containing the program ID for dimension 0
        pid_1 (list): List containing the program ID for dimension 1
        pid_2 (list): List containing the program ID for dimension 2
    """
    if test_pid_conds(conds, pid_0, pid_1, pid_2): print(txt)

def cdiv(a, b): 
    """
    Ceiling division - divides a by b and rounds up to the nearest integer.
    
    This is commonly used in GPU programming to calculate grid dimensions.
    
    Args:
        a (int): Numerator
        b (int): Denominator
        
    Returns:
        int: Ceiling division result
        
    Examples:
        >>> cdiv(10, 3)
        4
        >>> cdiv(10, 2)
        5
    """
    return (a + b - 1) // b


if __name__ == '__main__':
    # Test cases for the utility functions
    assert test_pid_conds('')
    assert test_pid_conds('>0', [1], [1])
    assert not test_pid_conds('>0', [0], [1])
    assert test_pid_conds('=0,=1', [0], [1], [0])
    assert cdiv(10,2)==5
    assert cdiv(10,3)==4