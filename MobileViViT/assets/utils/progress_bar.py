def progress_bar(current: int, total: int, bar_length: int = 25, description: str = '') -> str:

    """

    Display a progress bar inside a string.


    Parameters
    ----------
    current : int
        Current progress.

    total : int
        Total progress.

    bar_length : int, optional
        Length of the progress bar. The default value is 25.
        If bar_length is 0, only the percentage is displayed.

    description : str, optional
        Description of the progress bar. The default value is ''.


    Returns
    -------
    progress_bar : str
        Progress bar as string.

    """

    if not isinstance(current, int):
        raise TypeError("current must be an integer.")
    if current < 0:
        raise ValueError("current must be positive.")
    if not isinstance(total, int):
        raise TypeError("total must be an integer.")
    if total <= 0:
        raise ValueError("total must be positive.")
    if current > total:
        raise ValueError("current must be less than or equal to total.")
    if not isinstance(bar_length, int):
        raise TypeError("bar_length must be an integer.")
    if bar_length < 0:
        raise ValueError("bar_length must be positive.")
    if not isinstance(description, str):
        raise TypeError("description must be a string.")


    progress = current / total
    arrow = '=' * int(round(bar_length * progress))
    spaces = ' ' * (bar_length - len(arrow))
    percent = int(progress * 100)

    
    return f"{description}[{arrow + spaces}] {percent}%" if bar_length > 0 else f"{description}{percent}%"