"""Pruebas de la promoción de versiones al alias que usa el servicio."""

from types import SimpleNamespace

import pytest

from proyecto_mlops_equipo_9.modeling.promote import ETIQUETA_CRITERIO, promover


class ClienteFalso:
    """Imita las llamadas de MlflowClient que usa la promoción."""

    def __init__(self, aliases, criterio_por_version):
        self.aliases = dict(aliases)
        self.criterio = criterio_por_version
        self.etiquetas = {}

    def get_registered_model(self, nombre):
        return SimpleNamespace(aliases=dict(self.aliases))

    def get_model_version(self, nombre, version):
        return SimpleNamespace(run_id=f"run-{version}")

    def get_run(self, run_id):
        valor = self.criterio.get(run_id.removeprefix("run-"))
        etiquetas = {} if valor is None else {ETIQUETA_CRITERIO: valor}
        return SimpleNamespace(data=SimpleNamespace(tags=etiquetas))

    def set_registered_model_alias(self, nombre, alias, version):
        self.aliases[alias] = version

    def set_model_version_tag(self, nombre, version, clave, valor):
        self.etiquetas[(version, clave)] = valor


def test_promueve_la_version_candidata_y_deja_rastro():
    cliente = ClienteFalso({"candidato": "1"}, {"1": "True"})
    resultado = promover("modelo", responsable="equipo9", cliente=cliente)
    assert cliente.aliases["campeon"] == "1"
    assert resultado["cambio"] and resultado["anterior"] is None
    assert cliente.etiquetas[("1", "campeon_promovido_por")] == "equipo9"
    assert cliente.etiquetas[("1", "campeon_version_anterior")] == "ninguna"
    assert ("1", "campeon_desde_utc") in cliente.etiquetas


@pytest.mark.parametrize("valor", ["False", None])
def test_no_promueve_si_no_cumple_el_criterio(valor):
    cliente = ClienteFalso({"candidato": "2", "campeon": "1"}, {"2": valor})
    with pytest.raises(ValueError, match="no cumple el criterio"):
        promover("modelo", cliente=cliente)
    assert cliente.aliases["campeon"] == "1"
    assert not cliente.etiquetas


def test_version_explicita_permite_revertir():
    cliente = ClienteFalso({"candidato": "2", "campeon": "2"}, {"1": "True", "2": "True"})
    resultado = promover("modelo", version=1, responsable="equipo9", cliente=cliente)
    assert cliente.aliases["campeon"] == "1"
    assert resultado["anterior"] == "2"
    assert cliente.etiquetas[("1", "campeon_version_anterior")] == "2"


def test_no_cambia_nada_si_ya_es_campeon():
    cliente = ClienteFalso({"candidato": "1", "campeon": "1"}, {"1": "True"})
    assert promover("modelo", cliente=cliente)["cambio"] is False
    assert not cliente.etiquetas


def test_falla_si_no_existe_el_alias_de_origen():
    with pytest.raises(LookupError, match="@candidato"):
        promover("modelo", cliente=ClienteFalso({}, {}))
