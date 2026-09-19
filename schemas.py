"""Parameter Schemas with Validation

Based on patterns from Pydantic and MetaGPT.
"""

from pydantic import BaseModel, validator
from typing import List, Optional, Dict, Any
from enum import Enum


class GasType(str, Enum):
    """Supported gas types"""
    CO2 = "CO2"
    CH4 = "CH4"
    N2 = "N2"
    H2 = "H2"
    O2 = "O2"
    He = "He"
    Ar = "Ar"
    NH3 = "NH3"
    SO2 = "SO2"
    H2S = "H2S"


class GCMCParams(BaseModel):
    """GCMC simulation parameters"""
    material: str
    gas: GasType
    temperature: float = 298.0
    pressure_range: List[float] = [0.01, 10.0]
    force_field: str = "UFF"
    charge_method: str = "DDEC6"
    framework_dir: str = "./cifs"
    
    @validator('temperature')
    def validate_temperature(cls, v):
        if v < 0 or v > 1000:
            raise ValueError(f"Temperature must be between 0 and 1000 K, got {v}")
        return v
    
    @validator('pressure_range')
    def validate_pressure_range(cls, v):
        if len(v) != 2:
            raise ValueError("Pressure range must have exactly 2 values [min, max]")
        if v[0] >= v[1]:
            raise ValueError(f"Min pressure ({v[0]}) must be less than max ({v[1]})")
        if v[0] < 0:
            raise ValueError("Pressure cannot be negative")
        return v
    
    @validator('force_field')
    def validate_force_field(cls, v):
        allowed = ['UFF', 'DREIDING', 'MMFF94', 'OPLS-AA']
        if v not in allowed:
            raise ValueError(f"Force field must be one of {allowed}, got {v}")
        return v
    
    @validator('charge_method')
    def validate_charge_method(cls, v):
        allowed = ['DDEC6', 'CM5', 'REPEAT', 'Bader']
        if v not in allowed:
            raise ValueError(f"Charge method must be one of {allowed}, got {v}")
        return v


class CDFTParams(BaseModel):
    """cDFT simulation parameters"""
    material: str
    gas: GasType
    temperature: float = 298.0
    charge_method: str = "DDEC6"
    functional: str = "PBE"
    basis_set: str = "def2-SVP"
    
    @validator('functional')
    def validate_functional(cls, v):
        allowed = ['PBE', 'B3LYP', 'HSE06', 'SCAN']
        if v not in allowed:
            raise ValueError(f"Functional must be one of {allowed}, got {v}")
        return v
    
    @validator('basis_set')
    def validate_basis_set(cls, v):
        allowed = ['def2-SVP', 'def2-TZVP', 'def2-TZVPP', '6-31G*']
        if v not in allowed:
            raise ValueError(f"Basis set must be one of {allowed}, got {v}")
        return v


class DiffusionParams(BaseModel):
    """Diffusion calculation parameters"""
    material: str
    gas: GasType
    temperature: float = 298.0
    force_field: str = "UFF"
    n_molecules: int = 10
    n_steps: int = 1000000
    timestep: float = 0.5
    
    @validator('n_molecules')
    def validate_n_molecules(cls, v):
        if v < 1 or v > 1000:
            raise ValueError(f"Number of molecules must be between 1 and 1000, got {v}")
        return v
    
    @validator('timestep')
    def validate_timestep(cls, v):
        if v < 0.1 or v > 2.0:
            raise ValueError(f"Timestep must be between 0.1 and 2.0 fs, got {v}")
        return v


class PoreAnalysisParams(BaseModel):
    """Pore analysis parameters"""
    material: str
    probe_radius: float = 1.655  # N2 kinetic radius in Angstrom
    n_points: int = 100000
    
    @validator('probe_radius')
    def validate_probe_radius(cls, v):
        if v < 0.5 or v > 5.0:
            raise ValueError(f"Probe radius must be between 0.5 and 5.0 Angstrom, got {v}")
        return v


def validate_params(task_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate parameters for given task type"""
    try:
        if task_type == "gcmc":
            validated = GCMCParams(**params)
            return validated.dict()
        elif task_type == "cdft":
            validated = CDFTParams(**params)
            return validated.dict()
        elif task_type == "diffusion":
            validated = DiffusionParams(**params)
            return validated.dict()
        elif task_type == "pore_analysis":
            validated = PoreAnalysisParams(**params)
            return validated.dict()
        else:
            return params
    except Exception as e:
        raise ValueError(f"Parameter validation failed: {e}")


# 测试
if __name__ == "__main__":
    # 测试GCMC参数
    print("测试GCMC参数验证:")
    try:
        params = GCMCParams(
            material="MOF-5",
            gas="CO2",
            temperature=298,
            pressure_range=[0.01, 10]
        )
        print(f"✅ 有效参数: {params.dict()}")
    except Exception as e:
        print(f"❌ 验证失败: {e}")
    
    # 测试无效温度
    print("\n测试无效温度:")
    try:
        params = GCMCParams(
            material="MOF-5",
            gas="CO2",
            temperature=1500  # 超出范围
        )
        print(f"✅ 有效参数: {params.dict()}")
    except Exception as e:
        print(f"❌ 验证失败: {e}")
    
    # 测试无效气体
    print("\n测试无效气体:")
    try:
        params = GCMCParams(
            material="MOF-5",
            gas="XYZ",  # 不支持的气体
            temperature=298
        )
        print(f"✅ 有效参数: {params.dict()}")
    except Exception as e:
        print(f"❌ 验证失败: {e}")
