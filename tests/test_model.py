import torch

from picoret import PicoRet


def test_parameter_count_and_shape():
    model = PicoRet().eval()
    assert model.parameter_count == 817
    with torch.no_grad():
        output = model(torch.rand(1, 3, 31, 37))
    assert output.shape == (1, 3, 31, 37)
    assert torch.isfinite(output).all()
