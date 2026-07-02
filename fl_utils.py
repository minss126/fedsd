import torch
import copy

def init_server_optimizers(global_model):
    """
    서버 측 옵티마이저(Momentum 등)를 위한 변수 초기화
    """
    moment_first = {}
    moment_second = {}
    for key, param in global_model.named_parameters():
        moment_first[key] = torch.zeros_like(param.data)
        moment_second[key] = torch.zeros_like(param.data)
    return moment_first, moment_second

def aggregate_models(args, nets_this_round, fed_avg_freqs, global_w):
    """
    클라이언트 모델들을 가중 평균(Weighted Average)하여 집계하는 함수
    * 수정사항 1: LongTensor(num_batches_tracked)의 Float 연산 오류 방지
    * 수정사항 2: IndexError 방지를 위해 fed_avg_freqs 접근 방식 변경 (ID -> Index)
    """
    # 1. 첫 번째 클라이언트 모델을 기준으로 초기화
    client_ids = list(nets_this_round.keys())
    
    first_net_id = client_ids[0]
    first_net_para = nets_this_round[first_net_id].state_dict()
    
    # [수정됨] net_id가 아니라 0번(첫번째) 인덱스를 사용
    first_freq = fed_avg_freqs[0]
    
    new_w = {}
    
    # 첫 번째 모델 데이터 적재
    for key in first_net_para:
        param = first_net_para[key]
        if param.dtype == torch.long:
            new_w[key] = (param * first_freq).float()
        else:
            new_w[key] = param * first_freq

    # 2. 나머지 클라이언트 모델들 누적
    for i in range(1, len(client_ids)):
        net_id = client_ids[i]
        net_para = nets_this_round[net_id].state_dict()
        
        # [수정됨] net_id가 아니라 loop 인덱스 i를 사용
        freq = fed_avg_freqs[i]
        
        for key in new_w:
            param = net_para[key]
            if param.dtype == torch.long:
                new_w[key] += (param * freq).float()
            else:
                new_w[key] += param * freq

    # 3. 자료형 복구 (Float -> Long)
    for key in new_w:
        if 'num_batches_tracked' in key or new_w[key].dtype != global_w[key].dtype:
            if global_w[key].dtype == torch.long:
                new_w[key] = new_w[key].long()
            
    return new_w

def apply_server_side_optimization(args, global_w, old_w, nets_this_round, fed_avg_freqs, moment_first, moment_second):
    """
    서버 측 최적화 알고리즘 적용 (FedAvgM, FedAdam 등)
    """
    # FedAvg (기본) 인 경우 아무 작업 없이 반환
    if args.alg == 'fedavg' or args.alg == 'fedbyot' or args.alg == 'flocora':
        return global_w, moment_first, moment_second

    # FedAvgM (Server Momentum)
    if args.alg == 'fedavgM':
        beta = args.server_momentum
        for key in global_w:
            if 'num_batches_tracked' in key or 'running' in key:
                continue
            
            update_delta = global_w[key] - old_w[key]
            moment_first[key] = beta * moment_first[key] + update_delta
            global_w[key] = old_w[key] + moment_first[key]
            
        return global_w, moment_first, moment_second

    return global_w, moment_first, moment_second