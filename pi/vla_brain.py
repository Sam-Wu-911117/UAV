# vla_brain.py (執行於 Python 3.12)
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

app = FastAPI()
vla_device = torch.device("cpu")

print("⏳ [右腦] 正在加載 SmolVLA 模型至 CPU...")
try:
    vla_policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
    vla_policy.to(vla_device)
    vla_policy.eval()
    print("✅ [右腦] SmolVLA 模型加載完成，等待左腦呼叫！")
except Exception as e:
    vla_policy = None
    print(f"❌ [右腦] 加載失敗: {e}")

class InferenceRequest(BaseModel):
    text_command: str
    battery: float

@app.post("/vla/infer")
def infer_action(req: InferenceRequest):
    if vla_policy is None:
        return {"status": "error", "message": "VLA 模型未上線"}
        
    print(f"🧠 [右腦推理] 收到文字: '{req.text_command}', 電量: {req.battery}%")
    
    # 這裡未來會換成真實模型推理 vla_policy.select_action(...)
    mission_plan = [
        {"action_type": "TAKEOFF", "param": 5},
        {"action_type": "MOVE_FORWARD", "param": 3},
        {"action_type": "RTL", "param": 0}
    ]
    
    return {"status": "success", "mission_plan": mission_plan}

if __name__ == "__main__":
    # 右腦微服務運行於 Port 5001
    uvicorn.run(app, host="127.0.0.1", port=5001, log_level="warning")