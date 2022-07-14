import numpy as np


class GridGenerator:
  def __init__(self, height: int, width: int, kernel_size, stride=1):
    self.height = height
    self.width = width
    self.kernel_size = kernel_size  # (Kh, Kw)
    self.stride = stride  # (H, W)

  def createSamplingPattern(self):
    """
    :return: (1, H*Kh, W*Kw, (Lat, Lon)) sampling pattern
    """
    kerX, kerY = self.createKernel()  # (Kh, Kw)

    # create some values using in generating lat/lon sampling pattern
    rho = np.sqrt(kerX ** 2 + kerY ** 2)
    Kh, Kw = self.kernel_size
    # when the value of rho at center is zero, some lat values explode to `nan`.
    if Kh % 2 and Kw % 2:
      rho[Kh // 2][Kw // 2] = 1e-8

    nu = np.arctan(rho)
    cos_nu = np.cos(nu)
    sin_nu = np.sin(nu)

    stride_h, stride_w = self.stride, self.stride
    h_range = np.arange(0, self.height, stride_h)
    w_range = np.arange(0, self.width, stride_w)

    lat_range = ((h_range / self.height) - 0.5) * np.pi
    lon_range = ((w_range / self.width) - 0.5) * (2 * np.pi)

    # generate latitude sampling pattern
    lat = np.array([
      np.arcsin(cos_nu * np.sin(_lat) + kerY * sin_nu * np.cos(_lat) / rho) for _lat in lat_range
    ])  # (H, Kh, Kw)

    lat = np.array([lat for _ in lon_range])  # (W, H, Kh, Kw)
    lat = lat.transpose((1, 0, 2, 3))  # (H, W, Kh, Kw)

    # generate longitude sampling pattern
    lon = np.array([
      np.arctan(kerX * sin_nu / (rho * np.cos(_lat) * cos_nu - kerY * np.sin(_lat) * sin_nu)) for _lat in lat_range
    ])  # (H, Kh, Kw)

    lon = np.array([lon + _lon for _lon in lon_range])  # (W, H, Kh, Kw)
    lon = lon.transpose((1, 0, 2, 3))  # (H, W, Kh, Kw)

    # (radian) -> (index of pixel)
    lat = (lat / np.pi + 0.5) * self.height
    lon = ((lon / (2 * np.pi) + 0.5) * self.width) % self.width

    LatLon = np.stack((lat, lon))  # (2, H, W, Kh, Kw) = ((lat, lon), H, W, Kh, Kw)
    LatLon = LatLon.transpose((1, 2, 3, 4, 0))  # (H, Kh, W, Kw, 2) = (H, Kh, W, Kw, (lat, lon))

    H, W, Kh, Kw, d = LatLon.shape
    LatLon = LatLon.reshape((1, H, W, Kh*Kw, d))  # (1, H*Kh, W*Kw, 2)

    return LatLon

  def createKernel(self):
    """
    :return: (Ky, Kx) kernel pattern
    """
    Kh, Kw = self.kernel_size

    delta_lat = np.pi / self.height
    delta_lon = 2 * np.pi / self.width

    range_x = np.arange(-(Kw // 2), Kw // 2 + 1)
    if not Kw % 2:
      range_x = np.delete(range_x, Kw // 2)

    range_y = np.arange(-(Kh // 2), Kh // 2 + 1)
    if not Kh % 2:
      range_y = np.delete(range_y, Kh // 2)

    kerX = np.tan(range_x * delta_lon)
    kerY = np.tan(range_y * delta_lat) / np.cos(range_y * delta_lon)

    return np.meshgrid(kerX, kerY)  # (Kh, Kw)
