"""STL binário e pacote 3MF Core com unidades explícitas em milímetros."""

from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZIP_DEFLATED, ZipFile


def export_3mf(mesh, path):
    ns = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
    ET.register_namespace("", ns)

    def tag(name):
        return f"{{{ns}}}{name}"

    model = ET.Element(
        tag("model"),
        unit="millimeter",
        attrib={"{http://www.w3.org/XML/1998/namespace}lang": "pt-BR"},
    )
    resources = ET.SubElement(model, tag("resources"))
    obj = ET.SubElement(resources, tag("object"), id="1", type="model")
    geometry = ET.SubElement(obj, tag("mesh"))
    vertices = ET.SubElement(geometry, tag("vertices"))
    for vertex in mesh.vertices:
        ET.SubElement(
            vertices,
            tag("vertex"),
            **dict(zip(("x", "y", "z"), (format(float(v), ".9g") for v in vertex))),
        )
    triangles = ET.SubElement(geometry, tag("triangles"))
    for face in mesh.faces:
        ET.SubElement(
            triangles, tag("triangle"), **dict(zip(("v1", "v2", "v3"), map(str, face)))
        )
    build = ET.SubElement(model, tag("build"))
    ET.SubElement(build, tag("item"), objectid="1")
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/></Types>',
        )
        archive.writestr(
            "_rels/.rels",
            '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>',
        )
        archive.writestr(
            "3D/3dmodel.model",
            ET.tostring(model, encoding="utf-8", xml_declaration=True),
        )


def export_mesh(mesh, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stl, mf = directory / "modelo.stl", directory / "modelo.3mf"
    mesh.export(stl, file_type="stl")
    export_3mf(mesh, mf)
    return stl, mf
