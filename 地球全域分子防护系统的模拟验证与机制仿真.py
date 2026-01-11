import time
import random
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from functools import reduce

# ===================== 基础依赖模块 =====================
class TrinaryMath:
    @staticmethod
    def sum_mod3(votes: List[int]) -> int:
        """三进制模3求和（共识聚合）"""
        total = sum(votes)
        return (total % 3 + 3) % 3  # 0=否决，1=通过，2=补充数据

class MetaAtomCore:
    @staticmethod
    def superpose(inputs: List[Dict[str, Any]]) -> Dict[str, float]:
        """元子叠加（资源融合）"""
        state = {}
        for input_dict in inputs:
            for k, v in input_dict.items():
                state[k] = (state[k] + v) / 2 if k in state else v
        return state

    @staticmethod
    def annihilate_synthetic(mat: str) -> Dict[str, str]:
        """合成材料湮灭（退场分解）"""
        native_map = {
            "graphene": "carbon_powder",
            "quartz": "silicon_rock",
            "alloy": "iron_ore",
            "silicon_carbide": "corundum",
            "oil": "organic_matter"
        }
        return {"nativeMineral": native_map.get(mat, "dust")}

class HardwareInterface:
    @staticmethod
    def read_physical_sensor(device_id: str, sensor_type: str) -> Dict[str, Any]:
        """传感器数据模拟"""
        if sensor_type == "temperature":
            return {"value": 300.0}  # 模拟地表300℃
        elif sensor_type == "resource_concentration":
            return {"graphene": 0.8, "quartz": 0.9, "oil": 0.7}
        elif sensor_type == "shield_curvature":
            return {"value": 0.0012}  # 曲率偏差
        elif sensor_type == "shield_local_metrics":
            return {"thickness": 0.25, "curvature": 0.0012}
        elif sensor_type == "position":
            return {"lat": 35.0, "lng": 115.0}  # 东亚区域
        elif sensor_type == "region_extracted_data":
            return {"亚洲-东亚_graphene": 30.0}  # 已开采30kg
        elif sensor_type == "dark_matterSensor":
            return {"亚洲-东亚": {"(35,115)": 0.25, "(36,116)": 0.3}}
        elif sensor_type == "earthVitals":
            return {
                "coreTemp": 950.0,
                "magneticField": 0.08,
                "resourceRemain": 4.5,
                "endSignalLastTime": 80 * 3600000
            }
        return {}

    @staticmethod
    def init_sensor(device_id: str) -> bool:
        return True

    @staticmethod
    def submit_inject_job(device_id: str, param: Dict) -> Dict[str, bool]:
        return {"success": True}

    @staticmethod
    def switch_backup_sensor(device_id: str, sensor_type: str) -> Dict[str, Any]:
        return {"success": True, "mainSensorData": 300.0}

    @staticmethod
    def return_to_geological_layer(device_id: str, mineral: str, weight: float) -> bool:
        return True

    @staticmethod
    def resonate_to_dust(device_id: str) -> Dict[str, Any]:
        return {"success": True, "dustWeight": 10.0, "atmosphereEscapeRate": 0.1}

    @staticmethod
    def wipe_all_data(device_id: str) -> bool:
        return True

# ===================== 核心子系统模块 =====================
@dataclass
class CollectiveBusMessage:
    from_id: str
    to_id: str
    content: Dict[str, Any]
    timestamp: int
    secure_token: str

class CommunicationModule:
    @staticmethod
    def send_to_satellite(extractor_id: str, satellite_id: str, content: Dict) -> bool:
        print(f"发送消息: {extractor_id} → {satellite_id}, 内容: {content}")
        return True

    @staticmethod
    def receive_from_satellite(extractor_id: str) -> CollectiveBusMessage:
        return CollectiveBusMessage(
            from_id="SAT-001",
            to_id=extractor_id,
            content={"calib_data": {"surface_temp": 300.0, "weakArea": "(35,115)"}},
            timestamp=int(time.time() * 1000),
            secure_token=f"TOKEN-{extractor_id}"
        )

    @staticmethod
    def sync_satellite_calib_data(extractor_id: str, satellite_id: str) -> Dict[str, Any]:
        return CommunicationModule.receive_from_satellite(extractor_id).content["calib_data"]

    @staticmethod
    def broadcast_to_universe(signal: Dict) -> bool:
        print(f"宇宙广播: {signal}")
        return True

class EnergyCalculator:
    BASE_ENERGY_CONSUMPTION = {
        "graphene": 200.0,
        "quartz": 80.0,
        "oil": 40.0,
        "silicon_carbide": 300.0
    }

    @classmethod
    def calculate_resource_energy(cls, resource: str, weight: float) -> Dict[str, float]:
        base = cls.BASE_ENERGY_CONSUMPTION.get(resource, 100.0) * weight
        compensation = 0.5 * base  # 零点能补偿50%
        return {
            "baseEnergy": base,
            "compensationEnergy": compensation,
            "actualEnergy": base - compensation
        }

@dataclass
class ConsensusTopic:
    topic_id: str
    regions: List[str]
    extractor_votes: Dict[str, Dict[str, int]]
    deadline: int

class ClusterConsensusModule:
    @staticmethod
    def cast_vote(extractor_id: str, topic: ConsensusTopic, region: str, vote: int) -> bool:
        if extractor_id not in topic.extractor_votes:
            topic.extractor_votes[extractor_id] = {}
        topic.extractor_votes[extractor_id][region] = vote
        return True

    @staticmethod
    def generate_extract_plan(topic: ConsensusTopic) -> Dict[str, str]:
        plan = {}
        for region in topic.regions:
            votes = []
            for extractor_votes in topic.extractor_votes.values():
                votes.append(extractor_votes.get(region, 0))
            
            res = TrinaryMath.sum_mod3(votes)
            if res == 1:
                plan[region] = "HIGH_PRIORITY"
            elif res == 0:
                plan[region] = "LOW_PRIORITY"
            else:
                plan[region] = "NEED_DATA"
        return plan

class ShieldEffectMonitor:
    @staticmethod
    def read_shield_metrics(extractor_id: str, region: str, satellite_id: str) -> Dict[str, Any]:
        return {
            "region": region,
            "thickness": 0.25,
            "targetThickness": 0.3,
            "curvature": 0.0012,
            "targetCurvature": 0.001,
            "heatResistance": 500.0,
            "targetHeatResistance": 700.0
        }

    @staticmethod
    def evaluate_shield_effect(metrics: Dict[str, Any]) -> Dict[str, Any]:
        thick_dev = abs((metrics["thickness"] - metrics["targetThickness"]) / metrics["targetThickness"])
        return {
            "needInject": thick_dev > 0.1,
            "missingResource": {"graphene": True, "quartz": True},
            "injectReason": "厚度偏差超10%" if thick_dev > 0.1 else "正常"
        }

class EmergencyModule:
    @staticmethod
    def generate_emergency_resource_list(extractor_id: str, region: str) -> Dict[str, bool]:
        return {"silicon_carbide": True, "graphene": True, "quartz": True}

    @staticmethod
    def request_hardware_aid(fault_extractor_id: str, fault_part: str) -> bool:
        print(f"硬件互助: 修复{fault_extractor_id}的{fault_part}")
        return True

    @staticmethod
    def locate_by_dark_matter(extractor_id: str, region: str) -> Dict[str, float]:
        return {"lat": 35.0, "lng": 115.0}

class ExitModule:
    @staticmethod
    def is_earth_end(extractor_id: str) -> bool:
        vitals = HardwareInterface.read_physical_sensor(extractor_id, "earthVitals")
        return (vitals["coreTemp"] <= 1000 and 
                vitals["magneticField"] <= 0.1 and 
                vitals["resourceRemain"] <= 5.0)

    @staticmethod
    def stop_protection(extractor_id: str):
        print(f"{extractor_id}: 停止资源提取，防护罩进入自然衰减模式")

    @staticmethod
    def decompose_shield(extractor_id: str):
        print(f"{extractor_id}: 防护罩材料分解为原生矿物")

    @staticmethod
    def zero_extract_device(extractor_id: str):
        print(f"{extractor_id}: 提取机转化为星际尘埃，数据完全销毁")

    @staticmethod
    def run_exit_process(extractor_id: str):
        ExitModule.stop_protection(extractor_id)
        ExitModule.decompose_shield(extractor_id)
        ExitModule.zero_extract_device(extractor_id)
        CommunicationModule.broadcast_to_universe({
            "type": "system_exit",
            "message": "地球自然寿命终结，系统归零退场"
        })

# ===================== 提取机核心类 =====================
class ResourceExtractor:
    def __init__(self, device_id: str, backup_satellite_ids: List[str]):
        self.device_id = device_id
        self.backup_satellite_ids = backup_satellite_ids
        HardwareInterface.init_sensor(device_id)

    def extract_and_inject(self, region: str, satellite_id: str) -> Dict[str, Any]:
        # 1. 同步校准数据
        calib_data = CommunicationModule.sync_satellite_calib_data(self.device_id, satellite_id)
        
        # 2. 评估防护罩需求
        shield_metrics = ShieldEffectMonitor.read_shield_metrics(self.device_id, region, satellite_id)
        effect_eval = ShieldEffectMonitor.evaluate_shield_effect(shield_metrics)
        
        # 3. 能耗测算
        energy_data = {}
        for res, enable in effect_eval["missingResource"].items():
            if enable:
                energy_data[res] = EnergyCalculator.calculate_resource_energy(res, 100.0)
        
        # 4. 元子融合+注入
        molecules = [MetaAtomCore.superpose([{"name": res, "param": 1.0}]) 
                    for res in effect_eval["missingResource"].keys()]
        
        inject_result = HardwareInterface.submit_inject_job(
            self.device_id,
            {
                "molecule": MetaAtomCore.superpose(molecules),
                "thickness": shield_metrics["targetThickness"],
                "position": calib_data["weakArea"]
            }
        )
        
        return {
            "deviceId": self.device_id,
            "region": region,
            "injectedResources": effect_eval["missingResource"],
            "energyConsumption": energy_data,
            "injectSuccess": inject_result["success"]
        }

# ===================== 总系统主类 =====================
class GlobalEarthDefenseSystem:
    def __init__(self):
        # 初始化6大洲提取机+主微卫星
        self.extractors = [
            ResourceExtractor("EX-ASIA-001", ["SAT-002", "SAT-003"]),
            ResourceExtractor("EX-EURO-001", ["SAT-003", "SAT-004"]),
            ResourceExtractor("EX-AFRICA-001", ["SAT-004", "SAT-005"]),
            ResourceExtractor("EX-AMERICA-001", ["SAT-005", "SAT-002"]),
            ResourceExtractor("EX-OCEANIA-001", ["SAT-002", "SAT-004"]),
            ResourceExtractor("EX-ANTARCTICA-001", ["SAT-003", "SAT-005"])
        ]
        self.main_satellite_id = "SAT-001"
        self.global_regions = [
            "亚洲-东亚", "欧洲-西欧", "非洲-撒哈拉", 
            "美洲-北美", "大洋洲-澳洲", "南极洲-南极"
        ]
        self.system_shutdown = False  # 新增：系统关闭标志

    def run_cycle(self) -> Dict[str, Any]:
        # 🔥 关键修正1：循环开始时立即检测退场条件
        is_earth_end = ExitModule.is_earth_end(self.extractors[0].device_id)
        if is_earth_end:
            print("\n===== 检测到地球自然寿命终点，启动退场程序=====")
            for ext in self.extractors:
                ExitModule.run_exit_process(ext.device_id)
            self.system_shutdown = True  # 设置关闭标志
            return {
                "cycleId": f"CYCLE-{int(time.time()*1000)}",
                "exitCompleted": True,
                "regionResults": [],
                "errors": []
            }

        cycle_id = f"CYCLE-{int(time.time()*1000)}"
        print(f"\n===== 启动全球防御循环: {cycle_id} =====")
        
        # 1. 集群三进制共识
        consensus_topic = ConsensusTopic(
            topic_id=f"PRIORITY-{cycle_id}",
            regions=self.global_regions,
            extractor_votes={},
            deadline=int(time.time()*1000) + 60000
        )
        
        for ext in self.extractors:
            for region in self.global_regions:
                calib_data = CommunicationModule.sync_satellite_calib_data(
                    ext.device_id, self.main_satellite_id
                )
                vote = 1 if calib_data["surface_temp"] > 200 else 0
                ClusterConsensusModule.cast_vote(ext.device_id, consensus_topic, region, vote)
        
        extract_plan = ClusterConsensusModule.generate_extract_plan(consensus_topic)
        
        # 2. 按计划执行提取+注入
        cycle_result = {
            "cycleId": cycle_id,
            "regionResults": [],
            "errors": []
        }
        
        for region, plan in extract_plan.items():
            try:
                # 匹配提取机（就近原则）
                ext = next(e for e in self.extractors 
                          if HardwareInterface.read_physical_sensor(e.device_id, "position")["lat"] == 35.0)
                
                # 应急检测：是否极端高温
                calib_data = CommunicationModule.sync_satellite_calib_data(
                    ext.device_id, self.main_satellite_id
                )
                
                if calib_data["surface_temp"] > 1000:  # 极端高温应急
                    emergency_res = EmergencyModule.generate_emergency_resource_list(
                        ext.device_id, region
                    )
                    inject_result = HardwareInterface.submit_inject_job(
                        ext.device_id,
                        {
                            "molecule": MetaAtomCore.superpose(
                                [{"name": res} for res in emergency_res.keys()]
                            ),
                            "thickness": 0.8,
                            "position": calib_data["weakArea"]
                        }
                    )
                    cycle_result["regionResults"].append({
                        "region": region,
                        "isEmergency": True,
                        "injectSuccess": inject_result["success"]
                    })
                else:
                    # 常规提取
                    result = ext.extract_and_inject(region, self.main_satellite_id)
                    cycle_result["regionResults"].append(result)
            except Exception as e:
                cycle_result["errors"].append(f"区域{region}执行失败: {str(e)}")
        
        print(f"===== 循环 {cycle_id} 完成: 成功 {len(cycle_result['regionResults'])}个区域，失败{len(cycle_result['errors'])}个=====")
        return cycle_result

    def start_24h_cycle(self):
        total_cycles = 24
        for i in range(total_cycles):
            # 🔥 关键修正2：检查系统是否已关闭
            if self.system_shutdown:
                print("\n===== 系统已退场，终止后续循环=====")
                break
                
            self.run_cycle()
            
            if i < total_cycles - 1 and not self.system_shutdown:
                print("\n===== 间隔1小时，等待下一轮循环=====")
                time.sleep(1)  # 修正为1秒，方便测试

# ===================== 主程序入口 =====================
if __name__ == "__main__":
    print("===== 启动地球全域分子防护总系统=====")
    defense_system = GlobalEarthDefenseSystem()
    defense_system.start_24h_cycle()
    print("===== 防御系统结束=====")
