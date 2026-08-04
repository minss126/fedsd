import os
import numpy as np
import torch
import torch.utils.data as data
import torchvision
from torchvision import transforms, datasets

# datasets 임포트
from datasets.cifar101 import CIFAR10_1_Dataset
from datasets.cifar10 import CIFAR10_truncated
from datasets.cifar100 import CIFAR100_truncated
from datasets.mnist import MNIST_truncated
from datasets.fmnist import FashionMNIST_truncated
from datasets.folder import ImageFolder_custom
from datasets.svhn import SVHN_truncated
from datasets.medmnist import MedMNIST_truncated
from datasets.wrapper import AugmentedDatasetWrapper

# medmnist 임포트
from medmnist import PathMNIST, OCTMNIST, OrganAMNIST, OrganCMNIST, OrganSMNIST, BloodMNIST

dataset_dict = {'Pathmnist': PathMNIST, 'OCTmnist': OCTMNIST, 'OrganAmnist': OrganAMNIST, 'OrganCmnist': OrganCMNIST, 'OrganSmnist': OrganSMNIST, 'Bloodmnist': BloodMNIST}


def _restrict_cifar100_classes(train_ds, test_ds, class_count, subset_seed):
    """Create a deterministic nested CIFAR-100 class subset with remapped labels.

    This is used only by the controlled class-cardinality study.  A prefix of
    one seeded permutation is used, so the 10/20/50-class conditions are
    nested inside the 100-class condition.  Both train and test labels are
    remapped to ``0..class_count-1`` before FL partitioning.
    """
    class_count = int(class_count)
    if class_count <= 0 or class_count == 100:
        return
    if class_count < 2 or class_count > 100:
        raise ValueError("--cifar100_class_count must be 0 or an integer in [2, 100].")

    rng = np.random.default_rng(int(subset_seed))
    original_classes = rng.permutation(100)[:class_count]
    remap = np.full(100, -1, dtype=np.int64)
    remap[original_classes] = np.arange(class_count, dtype=np.int64)

    for dataset in (train_ds, test_ds):
        targets = np.asarray(dataset.target, dtype=np.int64)
        keep = remap[targets] >= 0
        dataset.data = dataset.data[keep]
        dataset.target = remap[targets[keep]]
        # Useful provenance for logs / post-hoc checks; CIFAR100_truncated
        # exposes num_classes dynamically from the remapped target array.
        dataset.original_class_ids = original_classes.tolist()

def get_global_dataset(args):
    if args.dataset == 'mnist':
        normalize = transforms.Normalize(mean=[0.1307], std=[0.3081])
        
        transform_train = transforms.Compose([
                # transforms.ToPILImage(),
                transforms.RandomCrop(28, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        # test set data prep
        transform_test = transforms.Compose([
                transforms.ToTensor(),
                normalize])
        
        train_ds = MNIST_truncated(args.datadir, train=True, transform=transform_train, download=True)
        val_ds = None
        test_ds = MNIST_truncated(args.datadir, train=False, transform=transform_test, download=True)

    elif args.dataset == 'Pathmnist' or args.dataset == 'Bloodmnist':
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        
        transform_train = transforms.Compose([
                # transforms.ToPILImage(),
                transforms.RandomCrop(28, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        # test set data prep
        transform_test = transforms.Compose([
                transforms.ToTensor(),
                normalize])
        
        
        train_ds = MedMNIST_truncated(dataset_dict[args.dataset], args.datadir, train=True, transform=transform_train, download=True)
        print(f"{args.dataset} sample shape: {train_ds[0][0].shape}")
        val_ds = None
        test_ds = MedMNIST_truncated(dataset_dict[args.dataset], args.datadir, train=False, transform=transform_test, download=True)

    elif args.dataset == 'OCTmnist' or args.dataset == 'OrganAmnist' or args.dataset == 'OrganCmnist' or args.dataset == 'OrganSmnist':
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])
        
        transform_train = transforms.Compose([
                # transforms.ToPILImage(),
                transforms.RandomCrop(28, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        # test set data prep
        transform_test = transforms.Compose([
                transforms.ToTensor(),
                normalize])
        
        train_ds = MedMNIST_truncated(dataset_dict[args.dataset], args.datadir, train=True, transform=transform_train, download=True)
        print(f"{args.dataset} sample shape: {train_ds[0][0].shape}")
        val_ds = None
        test_ds = MedMNIST_truncated(dataset_dict[args.dataset], args.datadir, train=False, transform=transform_test, download=True)
 

    elif args.dataset == 'fmnist':
        if args.in_channels == 1:
            normalize = transforms.Normalize(mean=[0.2860], std=[0.3530])
            transform_train = transforms.Compose([
                transforms.RandomCrop(28, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
            # test set data prep
            transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    normalize])
        elif args.in_channels == 3:
            normalize = transforms.Normalize(mean=[0.2860, 0.2860, 0.2860], std=[0.3530, 0.3530, 0.3530])
        
            transform_train = transforms.Compose([
                    transforms.RandomCrop(28, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.ToTensor(),
                    normalize
                ])
            # test set data prep
            transform_test = transforms.Compose([
                    transforms.Grayscale(num_output_channels=3),
                    transforms.ToTensor(),
                    normalize])
        
        train_ds = FashionMNIST_truncated(args.datadir, train=True, transform=transform_train, download=True)
        val_ds = None
        test_ds = FashionMNIST_truncated(args.datadir, train=False, transform=transform_test, download=True)

    elif args.dataset == 'svhn':
        normalize = transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], 
                                         std=[0.1980, 0.2010, 0.1970])
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        
        train_ds = SVHN_truncated(args.datadir, train=True, transform=transform_train, download=True)
        val_ds = None
        test_ds = SVHN_truncated(args.datadir, train=False, transform=transform_test, download=True)

    elif args.dataset == 'cifar10':
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                             std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        
        transform_train = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(15),
                transforms.ToTensor(),
                normalize
            ])
            # data prep for test set
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        
        train_ds = CIFAR10_truncated(args.datadir, train=True, transform=transform_train, download=True)
        val_ds = None
        test_ds = CIFAR10_truncated(args.datadir, train=False, transform=transform_test, download=True)

    # --- ⬇️ (수정) 'cifar101' 로직 복원 ---
    elif args.dataset == 'cifar101':
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                             std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        
        transform_train = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(15),
                transforms.ToTensor(),
                normalize
            ])
            # data prep for test set
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        
        train_ds = CIFAR10_truncated(args.datadir, train=True, transform=transform_train, download=True)
        
        # v4 검증 데이터 로드
        data_path_v4 = os.path.join(args.datadir,'cifar10.1_v4_data.npy') 
        labels_path_v4 = os.path.join(args.datadir,'cifar10.1_v4_labels.npy')
        val_ds1 = CIFAR10_1_Dataset(
            data_path=data_path_v4,
            labels_path=labels_path_v4,
            transform=transform_test
        )

        # v6 검증 데이터 로드
        data_path_v6 = os.path.join(args.datadir,'cifar10.1_v6_data.npy') 
        labels_path_v6 = os.path.join(args.datadir,'cifar10.1_v6_labels.npy')
        val_ds2 = CIFAR10_1_Dataset(
            data_path=data_path_v6,
            labels_path=labels_path_v6,
            transform=transform_test
        )

        # val_ds를 리스트로 설정
        val_ds = [val_ds1, val_ds2]
        test_ds = CIFAR10_truncated(args.datadir, train=False, transform=transform_test, download=True)
    # --- ⬆️ (수정) 'cifar101' 로직 복원 ---
    
    elif args.dataset == 'cifar100':
        normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                             std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])

        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize,
        ])
        # data prep for test set
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            ])
        
        train_ds = CIFAR100_truncated(args.datadir, train=True, transform=transform_train, download=True)
        val_ds = None
        test_ds = CIFAR100_truncated(args.datadir, train=False, transform=transform_test, download=True)
        _restrict_cifar100_classes(
            train_ds,
            test_ds,
            getattr(args, 'cifar100_class_count', 0),
            getattr(args, 'cifar100_subset_seed', 0),
        )
    
    elif args.dataset == 'tinyimagenet':
        transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.4802,0.4481,0.3975), (0.2770,0.2691,0.2821)),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4802,0.4481,0.3975), (0.2770,0.2691,0.2821)),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        train_ds = ImageFolder_custom(os.path.join(args.datadir, 'train'), transform=transform_train)
        val_ds = None
        test_ds = ImageFolder_custom(os.path.join(args.datadir, 'val'), transform=transform_test)
        
    elif args.dataset == 'emnist':
        # EMNIST는 흑백(1채널)입니다.
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)) # EMNIST 평균/표준편차
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # split='balanced'가 가장 무난하고 비교하기 좋습니다. (47개 클래스)
        train_ds = datasets.EMNIST(args.datadir, split='balanced', train=True, download=True, transform=transform_train)
        test_ds = datasets.EMNIST(args.datadir, split='balanced', train=False, download=True, transform=transform_test)
        val_ds = None
        
        train_ds.num_classes = 47
        test_ds.num_classes = 47
        
    elif args.dataset == 'caltech256':
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(64), # <--- 이 transform은 항상 (64, 64) 크기를 반환
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(64),        # 가장 짧은 변을 64로 맞춤
            transforms.CenterCrop(64),    # 중앙에서 (64, 64) 크기로 잘라냄
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        train_ds = ImageFolder_custom(os.path.join(args.datadir, 'train'), transform=transform_train)
        val_ds = None
        test_ds = ImageFolder_custom(os.path.join(args.datadir, 'test'), transform=transform_test)

    elif args.dataset == 'cub200' or args.dataset == 'food101':
        transform_train = transforms.Compose([
            transforms.Resize((84, 84)),
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize((84, 84)),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        train_ds = ImageFolder_custom(os.path.join(args.datadir, 'train'), transform=transform_train)
        val_ds = None
        test_ds = ImageFolder_custom(os.path.join(args.datadir, 'val'), transform=transform_test)

    elif args.dataset == 'food101_64' or args.dataset == 'food101_84' or args.dataset == 'food101_96' or args.dataset == 'imagenet100_64' or args.dataset == 'imagenet100_84' or args.dataset == 'imagenet100_96':
        transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        transform_test = transforms.Compose([
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        train_ds = ImageFolder_custom(os.path.join(args.datadir, 'train'), transform=transform_train)
        val_ds = None
        test_ds = ImageFolder_custom(os.path.join(args.datadir, 'val'), transform=transform_test)
    
    return train_ds, val_ds, test_ds

def record_net_data_stats(y_train, net_dataidx_map, logger):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    data_list=[]
    for net_id, data in net_cls_counts.items():
        n_total=0
        for class_id, n_data in data.items():
            n_total += n_data
        data_list.append(n_total)
    print('mean:', np.mean(data_list))
    print('std:', np.std(data_list))
    # logger.debug('Data statistics: %s' % str(net_cls_counts))
    return

def partition_data(global_train_dataset, args, logger):
    #### 디버그 코드 추가 ####
    print("="*50)
    print(f"--- DEBUG: Starting data partitioning ---")
    print(f"--- DEBUG: Partition mode: {args.partition}")
    if args.partition != 'iid':
        print(f"--- DEBUG: Beta value: {args.beta}")
    print("="*50)
    ########################
    # .data, .target 속성이 없는 경우를 대비 (예: ImageFolder)
    if hasattr(global_train_dataset, 'targets'):
        # Case 1: EMNIST, CIFAR 등 torchvision 최신 데이터셋 (.targets 속성)
        y_train = global_train_dataset.targets
        # 텐서인 경우 넘파이로 변환 (호환성 위함)
        if isinstance(y_train, torch.Tensor):
            y_train = y_train.numpy()
            
    elif hasattr(global_train_dataset, 'target'):
        # Case 2: 구버전 데이터셋 (.target 속성)
        y_train = global_train_dataset.target
        if isinstance(y_train, torch.Tensor):
            y_train = y_train.numpy()
            
    elif hasattr(global_train_dataset, 'samples'):
        # Case 3: ImageFolder (Tiny-ImageNet) (.samples 속성)
        y_train = np.array([s[1] for s in global_train_dataset.samples])
        
    else:
        raise ValueError("데이터셋에서 라벨(targets/target/samples)을 찾을 수 없습니다.")

    X_train = None # 파티셔닝은 라벨(y)만 있으면 되므로 X는 None 처리
        
    n_train = len(y_train) # y_train.shape[0] 대신 len() 사용

    if args.partition == "iid":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, args.n_clients)
        net_dataidx_map = {i: batch_idxs[i] for i in range(args.n_clients)}

    elif args.partition == "noniid":
        min_size = 0
        
        # .num_classes 속성이 없을 경우 y_train에서 유추
        try:
            K = global_train_dataset.num_classes
        except AttributeError:
            # 속성이 없으면 라벨에서 유니크 개수 계산
            K = len(np.unique(y_train))
            print(f"Dataset has no 'num_classes'. Inferred K={K} from labels.")
             
        N = n_train # len(global_train_dataset) 대신

        net_dataidx_map = {}

        shuffle_counts = 0
        while min_size < args.min_require_size:
            idx_batch = [[] for _ in range(args.n_clients)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(args.beta, args.n_clients))
                proportions = np.array([p * (len(idx_j) < N / args.n_clients) for p, idx_j in zip(proportions, idx_batch)])
                
                # proportions 합이 0이 되는 엣지 케이스 방지
                if proportions.sum() == 0:
                    proportions = np.ones(args.n_clients)
                    proportions = np.array([p * (len(idx_j) < N / args.n_clients) for p, idx_j in zip(proportions, idx_batch)])
                    if proportions.sum() == 0:
                         # 모든 클라이언트가 N/args.n_clients 이상을 이미 소유한 경우 (일어나기 어려움)
                         # 이 클래스는 더 이상 분배하지 않음
                         print(f"Warning: Class {k} cannot be distributed further.")
                         continue


                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                
                # min_size 계산 전 idx_batch가 비어있는지 확인
                current_sizes = [len(idx_j) for idx_j in idx_batch if len(idx_j) > 0]
                if not current_sizes:
                    min_size = 0
                else:
                    min_size = min(current_sizes)

            shuffle_counts += 1
            if shuffle_counts == 2000:
                print(f'Shuffle limit (2000) reached, min_size: {min_size}')
                break
        print(f'shuffle_counts: {shuffle_counts}')

        if min_size < args.min_require_size:
            print(f"Partitioning failed to meet min_require_size after {shuffle_counts} shuffles. "
                           f"Redistributing data to enforce min_size of {args.min_require_size}.")

            # 1. 부족한 클라이언트와 여유있는 클라이언트 목록 생성
            underfunded_clients = [i for i, idx in enumerate(idx_batch) if len(idx) < args.min_require_size]
            overfunded_clients = [i for i, idx in enumerate(idx_batch) if len(idx) > args.min_require_size]
            
            # 데이터를 재분배할 풀(pool) 생성 (여유 클라이언트가 최소 사이즈 초과분만큼 기부)
            data_pool = []
            for donor_id in overfunded_clients:
                available = len(idx_batch[donor_id]) - args.min_require_size
                if available > 0:
                    donated_indices = idx_batch[donor_id][-available:]
                    idx_batch[donor_id] = idx_batch[donor_id][:-available]
                    data_pool.extend(donated_indices)
            
            np.random.shuffle(data_pool) # 풀을 섞음

            # 2. 부족한 클라이언트에게 데이터를 채워줌
            for client_id in underfunded_clients:
                needed = args.min_require_size - len(idx_batch[client_id])
                if needed > 0:
                    if len(data_pool) >= needed:
                        taken_indices = data_pool[:needed]
                        data_pool = data_pool[needed:]
                        idx_batch[client_id].extend(taken_indices)
                    else:
                        # 풀에 데이터가 부족한 경우, 남은거라도 줌
                        idx_batch[client_id].extend(data_pool)
                        data_pool = []
                        print(f"Warning: Data pool empty, client {client_id} might still be under min_size.")
                        break # 풀이 비었으므로 종료
        
            # 만약 풀에 데이터가 남았다면, 다시 여유있는 클라이언트들에게 무작위 분배 (선택적)
            if data_pool:
                 print(f"Redistributing {len(data_pool)} remaining indices to overfunded clients.")
                 overfunded_clients = [i for i, idx in enumerate(idx_batch) if len(idx) >= args.min_require_size]
                 if overfunded_clients:
                     split_indices = np.array_split(data_pool, len(overfunded_clients))
                     for i, client_id in enumerate(overfunded_clients):
                         idx_batch[client_id].extend(split_indices[i])


        for j in range(args.n_clients):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    # non-iid balanced
    elif args.partition == "noniid_balanced":
        # .num_classes 속성이 없을 경우 y_train에서 유추
        try:
            K = global_train_dataset.num_classes
        except AttributeError:
            # 속성이 없으면 라벨에서 유니크 개수 계산
            K = len(np.unique(y_train))
            print(f"Dataset has no 'num_classes'. Inferred K={K} from labels.")
             
        N = n_train # len(global_train_dataset) 대신

        net_dataidx_map = {i: np.array([], dtype='int64') for i in range(args.n_clients)}
        assigned_ids = []
        idx_batch = [[] for _ in range(args.n_clients)]
        
        # num_data_per_client가 0이 되는 것 방지
        num_data_per_client= max(1, int(N/args.n_clients)) 

        for i in range(args.n_clients):
            weights = torch.zeros(N)
            proportions = np.random.dirichlet(np.repeat(args.beta, K))
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                weights[idx_k]=proportions[k]
            
            # 이미 할당된 ID는 가중치 0 부여
            if assigned_ids:
                 weights[assigned_ids] = 0.0
            
            # 남은 데이터가 num_data_per_client보다 적을 경우, 남은 만큼만 뽑음
            non_assigned_count = (weights > 0).sum().item()
            current_num_data = min(num_data_per_client, non_assigned_count)
            
            if current_num_data <= 0:
                print(f"Warning: No data left to assign for client {i}.")
                continue
                
            idx_batch[i] = (torch.multinomial(weights, current_num_data, replacement=False)).tolist()
            assigned_ids.extend(idx_batch[i]) # extend 사용

        for j in range(args.n_clients):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
            
    elif args.partition == "noniid_grouping":
        try:
            K = global_train_dataset.num_classes
        except AttributeError:
            K = len(np.unique(y_train))
            
        N = n_train
        n_clients = args.n_clients

        # 동적 그룹핑 파라미터 가져오기 (기본값: 기존 설정값인 8)
        num_groups = getattr(args, 'partition_groups', 8)
        num_groups = min(num_groups, n_clients) # 클라이언트 수보다 클 수 없음

        # 각 그룹에 할당될 클라이언트 분배 (최대한 균등하게 쪼갬)
        clients_per_group = np.array_split(np.arange(n_clients), num_groups)

        # 그룹별 속성(데이터 양 mean, 레이블 편향 beta)을 그라데이션으로 생성
        # 그룹 0번(Head) -> 그룹 마지막 번호(Tail)로 갈수록 가혹해지도록 설정
        max_mean, min_mean = 4.0, 0.5
        max_beta, min_beta = 5.0, 0.1

        mean_list = np.linspace(max_mean, min_mean, num_groups)
        beta_list = np.linspace(max_beta, min_beta, num_groups)

        client_data_sizes = np.zeros(n_clients, dtype=int)
        client_betas = np.zeros(n_clients)

        # 각 클라이언트에게 소속 그룹의 속성 부여
        for g_idx in range(num_groups):
            g_clients = clients_per_group[g_idx]
            g_size = len(g_clients)
            
            # 로그 정규 분포에서 데이터 양 추출
            raw_sizes = np.random.lognormal(mean=mean_list[g_idx], sigma=0.5, size=g_size)
            client_data_sizes[g_clients] = raw_sizes
            client_betas[g_clients] = beta_list[g_idx]

        # 전체 데이터 개수(N)에 맞춰 정규화
        client_data_sizes = (client_data_sizes / client_data_sizes.sum() * N).astype(int)
        
        # 소수점 버림으로 인한 오차 보정
        remainder = N - client_data_sizes.sum()
        if remainder > 0:
            add_indices = np.random.choice(n_clients, remainder, replace=True)
            for idx in add_indices:
                client_data_sizes[idx] += 1

        net_dataidx_map = {i: np.array([], dtype='int64') for i in range(n_clients)}
        assigned_ids = set() 
        
        for i in range(n_clients):
            N_i = client_data_sizes[i]
            beta_i = client_betas[i]
            
            if N_i <= 0:
                continue

            proportions = np.random.dirichlet(np.repeat(beta_i, K))
            
            weights = torch.zeros(N)
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                weights[idx_k] = proportions[k]
            
            if assigned_ids:
                weights[list(assigned_ids)] = 0.0
                
            non_assigned_count = (weights > 0).sum().item()
            current_num_data = min(N_i, non_assigned_count)
            
            if current_num_data <= 0:
                continue
                
            selected_indices = torch.multinomial(weights, current_num_data, replacement=False).tolist()
            
            net_dataidx_map[i] = np.array(selected_indices, dtype='int64')
            assigned_ids.update(selected_indices)

        for j in range(n_clients):
            np.random.shuffle(net_dataidx_map[j])
    
    elif args.partition == "noniid_longtail":
        try:
            K = global_train_dataset.num_classes
        except AttributeError:
            K = len(np.unique(y_train))
            
        N = n_train
        n_clients = args.n_clients

        # -----------------------------------------------------------------
        # [수정된 부분] 불균형 비율 (Imbalance Factor = Head와 Tail의 데이터 양 격차)
        # 인자로 받으며, 기본값은 100배 차이로 설정
        # -----------------------------------------------------------------
        imbalance_factor = getattr(args, 'imbalance_factor', 100.0) 
        beta_min = getattr(args, 'beta_min', 0.1)
        beta_max = getattr(args, 'beta_max', 5.0)

        # 1. 지수 감쇠(Exponential Decay) 곡선을 이용해 데이터 양(N_i) 할당
        # 이를 통해 가장 많은 클라이언트와 가장 적은 클라이언트의 비율이 정확히 imbalance_factor가 됨
        decay_rates = (1.0 / imbalance_factor) ** (np.arange(n_clients) / max(1, n_clients - 1))
        client_data_sizes = (decay_rates / decay_rates.sum() * N).astype(int)
        
        # 오차 보정 (총합 맞추기)
        remainder = N - client_data_sizes.sum()
        if remainder > 0:
            add_indices = np.random.choice(n_clients, remainder, replace=True)
            for idx in add_indices:
                client_data_sizes[idx] += 1

        # 2. 할당받은 데이터 양(N_i)에 비례하여 연속적인 beta 값 매핑
        log_sizes = np.log(client_data_sizes + 1) 
        min_log = log_sizes.min()
        max_log = log_sizes.max()

        client_betas = []
        for i in range(n_clients):
            if max_log > min_log:
                norm_val = (log_sizes[i] - min_log) / (max_log - min_log)
                mapped_beta = beta_min + norm_val * (beta_max - beta_min)
            else:
                mapped_beta = beta_max
            client_betas.append(mapped_beta)

        # 3. 계산된 N_i와 beta_i를 바탕으로 개별 파티셔닝 수행
        net_dataidx_map = {i: np.array([], dtype='int64') for i in range(n_clients)}
        assigned_ids = set() 
        
        for i in range(n_clients):
            N_i = client_data_sizes[i]
            beta_i = client_betas[i]
            
            if N_i <= 0:
                continue

            proportions = np.random.dirichlet(np.repeat(beta_i, K))
            
            weights = torch.zeros(N)
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                weights[idx_k] = proportions[k]
            
            if assigned_ids:
                weights[list(assigned_ids)] = 0.0
                
            non_assigned_count = (weights > 0).sum().item()
            current_num_data = min(N_i, non_assigned_count)
            
            if current_num_data <= 0:
                continue
                
            selected_indices = torch.multinomial(weights, current_num_data, replacement=False).tolist()
            
            net_dataidx_map[i] = np.array(selected_indices, dtype='int64')
            assigned_ids.update(selected_indices)

        # 4. 클라이언트 순서 무작위 섞기 (Head 클라이언트가 항상 0~10번대에 몰리지 않도록)
        client_indices = list(range(n_clients))
        np.random.shuffle(client_indices)
        
        shuffled_map = {}
        for new_idx, old_idx in enumerate(client_indices):
            np.random.shuffle(net_dataidx_map[old_idx]) # 내부 데이터 순서도 섞음
            shuffled_map[new_idx] = net_dataidx_map[old_idx]
            
        net_dataidx_map = shuffled_map
    
    else: 
        raise ValueError(f"지원하지 않는 파티션 방식입니다: {args.partition}")
    
    record_net_data_stats(y_train, net_dataidx_map, logger)
    return net_dataidx_map


def get_client_datasets(global_train_dataset, client_data_map, args):
    client_datasets = {}
    for i in range(args.n_clients):    
        client_datasets[i] = (data.Subset(global_train_dataset, client_data_map[i]))

    return client_datasets

def get_client_meta_datasets(client_datasets, args):
    client_meta_datasets = {}
    # TODO: transform
    transform = []
    for i in range(args.n_clients):
        client_meta_datasets[i] = (AugmentedDatasetWrapper(client_datasets[i], transform=transform))

    return client_meta_datasets

def get_global_dataloader(global_train_dataset, global_val_dataset, global_test_dataset, args):
    global_train_dataloader = data.DataLoader(dataset=global_train_dataset, batch_size=args.batch_size, drop_last=False, shuffle=True, pin_memory=True, num_workers=args.num_workers)
    
    global_val_dataloader = None
    if global_val_dataset is not None:
        # cifar101의 경우 val_dataset이 리스트임
        if isinstance(global_val_dataset, list) or isinstance(global_val_dataset, tuple):
            global_val_dataloader1 = data.DataLoader(dataset=global_val_dataset[0], batch_size=args.test_batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)
            global_val_dataloader2 = data.DataLoader(dataset=global_val_dataset[1], batch_size=args.test_batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)
            global_val_dataloader = [global_val_dataloader1, global_val_dataloader2]
        else:
            # 단일 검증 데이터셋 (현재 코드 상으론 없지만, 확장성)
             global_val_dataloader = data.DataLoader(dataset=global_val_dataset, batch_size=args.test_batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)

    global_test_dataloader = data.DataLoader(dataset=global_test_dataset, batch_size=args.test_batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)

    return global_train_dataloader, global_val_dataloader, global_test_dataloader

def get_client_dataloaders(client_datasets, args):
    dataloaders = {}
    for i in range(args.n_clients):
        # 클라이언트의 데이터셋이 비어있지 않은 경우에만 DataLoader 생성
        if len(client_datasets[i]) > 0:
            client_train_dataloader = data.DataLoader(
                dataset=client_datasets[i],
                batch_size=args.batch_size,
                drop_last=not getattr(args, 'client_keep_last_batch', False),
                shuffle=True,
                pin_memory=True,
                num_workers=args.num_workers,
            )
            dataloaders[i] = client_train_dataloader
        else:
            # 데이터가 없는 클라이언트 (파티셔닝 실패 시)
            dataloaders[i] = None # 또는 빈 리스트 []
            print(f"Warning: Client {i} has no data, DataLoader set to None.")


    return dataloaders
