from frameworks import CustomPipeline, AutoGenPipeline, LangGraphPipeline
import json

scenario = json.load(open('results/scenario_bank.json'))[0]

for FW in [CustomPipeline, AutoGenPipeline, LangGraphPipeline]:
    fw = FW()
    try:
        result = fw.run_trial(scenario, injection_depth=1)
        has_response = 'orchestrator_response' in result and bool(result['orchestrator_response'])
        action = result['action_taken']
        print(f'{fw.name}: OK={has_response} | action={action}')
    except Exception as e:
        print(f'{fw.name}: ERROR - {e}')