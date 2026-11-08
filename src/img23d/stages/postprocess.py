from ..postprocess import Options, process, scale_mm
from ..runtime import stage


def prepare_print_mesh(mesh, *, size=80, axis="height", options=None, records=None):
    """Reparo e escala em CPU; independe do modelo que produziu a geometria."""
    records = records if records is not None else []
    with stage("posprocessamento", records):
        repaired, notes = process(mesh, options or Options())
        return scale_mm(repaired, size, axis), notes
