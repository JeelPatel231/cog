from collections.abc import AsyncIterator


class LastValueIterator[T]:
  def __init__(self, iterator: AsyncIterator[T]) -> None:
    self.__iterator = iterator
    self.__last_value = None

  @property
  def last(self) -> T:
    return_val = self.__last_value
    assert return_val is not None
    return return_val
  
  def __aiter__(self):
      return self

  async def __anext__(self) -> T:
      value = await self.__iterator.__anext__()
      self.__last_value = value
      return value