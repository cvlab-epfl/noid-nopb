import torch


def get_coord_grid(x_size: int, y_size: int, device=None) -> torch.Tensor:
    """Creates a coordinate grid of size (x_size, y_size).
    
    Args:
        x_size: Width of the grid
        y_size: Height of the grid 
        device: PyTorch device to place tensor on

    Returns:
        Coordinate grid tensor of shape (y_size, x_size, 2) containing (x,y) coordinates
    """
    xs = torch.arange(0, x_size, device=device)
    ys = torch.arange(0, y_size, device=device)
    x, y = torch.meshgrid(xs, ys)
    
    coord_grid = torch.stack([x, y]).permute(2, 1, 0)
    
    return coord_grid.float()


def reconstruct_from_motion_offset(
    hm: torch.Tensor,
    offset: torch.Tensor = None,
    ksize: int = 3,
    expe_weight: float = 0.5,
    shift: float = -10.0,
    slope: float = 4.0
) -> torch.Tensor:
    """Reconstructs a heatmap by applying motion offsets.

    Takes a heatmap and optional motion offset field and reconstructs the heatmap by
    propagating values according to the offsets. Uses a windowed reconstruction with
    exponential weighting based on distance.

    Args:
        hm: Input heatmap tensor of shape (B, C, H, W)
        offset: Optional motion offset tensor of shape (B, H, W, 2) 
        ksize: Size of reconstruction window (must be odd)
        expe_weight: Weight for exponential distance weighting
        shift: Shift parameter for distance weighting
        slope: Slope parameter for distance weighting

    Returns:
        Reconstructed heatmap tensor of shape (B, C, H, W)
    """
    assert ksize % 2 == 1, "Reconstruction window size must be odd"
    
    B, C, H, W = hm.size()

    if offset is not None:
        B_o, H_o, W_o, C_o = offset.size()
        assert B == B_o, "Batch sizes must match"
        assert C_o == 2, "Offset must have 2 channels (x,y)"
        assert H == H_o, "Heights must match" 
        assert W == W_o, "Widths must match"
    
    # Generate coordinate grid
    coord_grid = get_coord_grid(W, H, hm.device)
    coord_grid = coord_grid.repeat(B, 1, 1, 1)
    new_coord = coord_grid.clone()
    
    # Apply offsets if provided
    updated_coord = coord_grid + offset if offset is not None else coord_grid
    
    # Setup reconstruction window
    kernel_h, kernel_w = ksize, ksize
    stride = 1
    padding = (kernel_w//2, kernel_w//2, kernel_h//2, kernel_h//2)

    # Reshape coordinates for windowed computation
    new_coord_u = new_coord.permute(0, 3, 1, 2).unsqueeze(4).unsqueeze(5)
    updated_coord_u = torch.nn.functional.pad(
        updated_coord.permute(0, 3, 1, 2), padding
    ).unfold(2, kernel_h, stride).unfold(3, kernel_w, stride)
    
    # Compute distance-based weights
    distance = -(torch.sqrt(torch.clamp(
        ((new_coord_u - updated_coord_u)**2).sum(dim=1, keepdim=True),
        min=1e-8
    )) * slope * expe_weight + shift)
    distance = torch.exp(distance)
    distance = distance / (distance + 1)

    # Apply windowed reconstruction
    hm_u = torch.nn.functional.pad(hm, padding).unfold(2, kernel_h, stride).unfold(3, kernel_w, stride)
    rec = (hm_u * distance).sum(dim=(4, 5))

    return rec