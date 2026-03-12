from cchess_alphazero.agent.backends import configure_backend_session


def set_session_config(per_process_gpu_memory_fraction=None, allow_growth=None, device_list='0'):
    """

    :param allow_growth: When necessary, reserve memory
    :param float per_process_gpu_memory_fraction: specify GPU memory usage as 0 to 1

    :return:
    """
    return configure_backend_session(
        per_process_gpu_memory_fraction=per_process_gpu_memory_fraction,
        allow_growth=allow_growth,
        device_list=device_list,
    )
