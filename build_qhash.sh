#!/bin/bash

# QHash Miner Build Script
# Автоматическая сборка майнера с поддержкой CUDA и cuQuantum
#
# Использование: ./build_qhash.sh [clean|rebuild|install]
#   clean   - очистить сборку
#   rebuild - полная пересборка (clean + build)
#   install - установить после сборки

set -e  # Выход при ошибке

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Функции логирования
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Автоопределение путей CUDA
detect_cuda_path() {
    local cuda_paths=(
        "/usr/local/cuda"
        "/opt/cuda"
        "/usr/cuda"
        "/usr/bin"
        "/usr"
    )
    
    for path in "${cuda_paths[@]}"; do
        if [[ -d "$path" && -f "$path/bin/nvcc" ]]; then
            echo "$path"
            return 0
        fi
    done
    
    # Проверяем PATH
    if command -v nvcc >/dev/null 2>&1; then
        nvcc_path=$(which nvcc)
        echo "${nvcc_path%/bin/nvcc}"
        return 0
    fi
    
    return 1
}

# Конфигурация путей
setup_paths() {
    log_info "Настройка путей..."
    
    # CUDA Path
    if [[ -z "$CUDA_PATH" ]]; then
        if CUDA_PATH=$(detect_cuda_path); then
            log_success "CUDA найден: $CUDA_PATH"
        else
            log_error "CUDA не найден! Установите CUDA Toolkit или задайте CUDA_PATH"
            exit 1
        fi
    fi
    export CUDA_PATH
    
    # cuQuantum Path - используем указанный пользователем путь
    if [[ -z "$CUQUANTUM_ROOT" ]]; then
        CUQUANTUM_ROOT="/home/henuzzz/cuquantum-linux-x86_64-25.06.0.10_cuda12-archive"
    fi
    
    if [[ ! -d "$CUQUANTUM_ROOT" ]]; then
        log_error "cuQuantum не найден по пути: $CUQUANTUM_ROOT"
        log_info "Проверьте путь или задайте переменную CUQUANTUM_ROOT"
        exit 1
    fi
    export CUQUANTUM_ROOT
    
    # Библиотечные пути - настраиваем в зависимости от расположения CUDA
    if [[ "$CUDA_PATH" == "/usr" ]]; then
        # CUDA установлен в системные пути
        export CUSTATEVEC_INCLUDES="-I/usr/include -I$CUQUANTUM_ROOT/include"
        export CUSTATEVEC_LDFLAGS="-L/usr/lib/x86_64-linux-gnu -L$CUQUANTUM_ROOT/lib -lcustatevec -lcudart -lcurl -lssl -lcrypto -ljansson -levent -lzmq"
        export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$CUQUANTUM_ROOT/lib:$LD_LIBRARY_PATH"
    else
        # CUDA установлен в отдельную директорию
        export CUSTATEVEC_INCLUDES="-I$CUDA_PATH/include -I$CUQUANTUM_ROOT/include"
        export CUSTATEVEC_LDFLAGS="-L$CUDA_PATH/lib64 -L$CUQUANTUM_ROOT/lib -lcustatevec -lcudart -lcurl -lssl -lcrypto -ljansson -levent -lzmq"
        export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$CUQUANTUM_ROOT/lib:$LD_LIBRARY_PATH"
    fi
    
    log_success "Пути настроены:"
    log_info "  CUDA_PATH: $CUDA_PATH"
    log_info "  CUQUANTUM_ROOT: $CUQUANTUM_ROOT"
}

# Проверка зависимостей
check_dependencies() {
    log_info "Проверка зависимостей..."
    
    # Проверяем базовые инструменты
    local tools=("gcc" "g++" "make" "autoconf" "automake" "pkg-config")
    for tool in "${tools[@]}"; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            log_error "Не найден: $tool"
            log_info "Установите: sudo apt-get install build-essential autoconf automake libtool pkg-config"
            exit 1
        fi
    done
    
    # Проверяем libtool отдельно (может быть libtool или libtoolize)
    if ! command -v "libtool" >/dev/null 2>&1 && ! command -v "libtoolize" >/dev/null 2>&1; then
        log_error "Не найден: libtool или libtoolize"
        log_info "Установите: sudo apt-get install libtool"
        exit 1
    fi
    
    # Проверяем CUDA
    local nvcc_found=false
    if [[ -f "$CUDA_PATH/bin/nvcc" ]]; then
        nvcc_found=true
    elif [[ "$CUDA_PATH" == "/usr" && -f "/usr/bin/nvcc" ]]; then
        nvcc_found=true
    fi
    
    if [[ "$nvcc_found" == "false" ]]; then
        log_error "CUDA nvcc не найден: $CUDA_PATH/bin/nvcc"
        log_info "Найденный nvcc: $(which nvcc 2>/dev/null || echo 'не найден')"
        exit 1
    fi
    
    # Проверяем cuQuantum
    if [[ ! -f "$CUQUANTUM_ROOT/include/custatevec.h" ]]; then
        log_error "cuQuantum заголовки не найдены: $CUQUANTUM_ROOT/include/custatevec.h"
        exit 1
    fi
    
    if [[ ! -f "$CUQUANTUM_ROOT/lib/libcustatevec.so" ]]; then
        log_error "cuQuantum библиотека не найдена: $CUQUANTUM_ROOT/lib/libcustatevec.so"
        exit 1
    fi
    
    # Проверяем системные библиотеки
    local libs=("libssl-dev" "libcurl4-openssl-dev" "libjansson-dev" "libevent-dev" "libzmq3-dev")
    for lib in "${libs[@]}"; do
        if ! dpkg -l | grep -q "$lib"; then
            log_warning "Возможно не установлен: $lib"
        fi
    done
    
    log_success "Зависимости проверены"
}

# Очистка сборки
clean_build() {
    log_info "[🧹] Очистка следов от старой сборки..."
    
    if [[ -f "Makefile" ]]; then
        make clean || true
    fi
    
    rm -rf autom4te.cache
    rm -f Makefile.in aclocal.m4 configure
    rm -f config.log config.status
    rm -f cpuminer-config.h cpuminer-config.h.in
    rm -f qubitcoin-miner
    
    log_success "[🧹] Сборка очищена"
}

# Настройка автотулов
setup_autotools() {
    log_info "[🔨] Настройка autotools..."
    
    if [[ ! -f "configure.ac" && ! -f "configure.in" ]]; then
        log_error "Не найден configure.ac или configure.in"
        exit 1
    fi
    
    # Генерируем автотулы
    if [[ -f "autogen.sh" ]]; then
        chmod +x autogen.sh
        ./autogen.sh
    else
        autoreconf -fiv
    fi
    
    log_success "[🔨] Autotools настроены"
}

# Конфигурация сборки
configure_build() {
    log_info "[🔨] Конфигурация сборки..."
    
    # Флаги компилятора
    export CFLAGS="-O3 -march=native -fomit-frame-pointer"
    export CXXFLAGS="-O3 -march=native -fomit-frame-pointer"
    
    # LDFLAGS в зависимости от расположения CUDA
    if [[ "$CUDA_PATH" == "/usr" ]]; then
        export LDFLAGS="-Wl,-rpath,/usr/lib/x86_64-linux-gnu -Wl,-rpath,$CUQUANTUM_ROOT/lib"
    else
        export LDFLAGS="-Wl,-rpath,$CUDA_PATH/lib64 -Wl,-rpath,$CUQUANTUM_ROOT/lib"
    fi
    
    # Конфигурируем с поддержкой CUDA и cuQuantum
    ./configure \
        --with-curl=/usr \
        --with-crypto=/usr \
        CUSTATEVEC_INCLUDES="$CUSTATEVEC_INCLUDES" \
        CUSTATEVEC_LDFLAGS="$CUSTATEVEC_LDFLAGS" \
        LIBS="-lcurl -lssl -lcrypto -ljansson -levent -lzmq -lz" \
        CFLAGS="$CFLAGS" \
        CXXFLAGS="$CXXFLAGS" \
        LDFLAGS="$LDFLAGS"
    
    log_success "[🔨] Конфигурация завершена"
}

# Сборка
build_project() {
    log_info "[🔨] Сборка проекта..."
    
    # Определяем количество ядер для параллельной сборки
    local num_cores
    if command -v nproc >/dev/null 2>&1; then
        num_cores=$(nproc)
    else
        num_cores=4
    fi
    
    log_info "[🔨] Используется $num_cores ядер для сборки"
    
    # Собираем
    make -j"$num_cores"
    
    if [[ -f "qubitcoin-miner" ]]; then
        log_success "[✅] Сборка завершена успешно!"
        
        # Показываем информацию о бинарном файле
        log_info "[ℹ️] Информация о собранном файле:"
        ls -lh qubitcoin-miner
        file qubitcoin-miner
        
        # Проверяем зависимости
        log_info "[🔨] Зависимости:"
        ldd qubitcoin-miner | grep -E "(cuda|custatevec)" || log_warning "CUDA/cuQuantum зависимости не найдены в ldd"
        
    else
        log_error "Сборка не удалась - исполняемый файл не создан"
        exit 1
    fi
}

# Установка (опционально)
install_binary() {
    log_info "Установка..."
    
    local install_dir="/usr/local/bin"
    if [[ ! -w "$install_dir" ]]; then
        log_warning "Нет прав записи в $install_dir, используйте sudo"
        sudo cp qubitcoin-miner "$install_dir/"
    else
        cp qubitcoin-miner "$install_dir/"
    fi
    
    log_success "Установлено в $install_dir/qubitcoin-miner"
}

# Тестирование
test_build() {
    log_info "Тестирование сборки..."
    
    if [[ ! -f "qubitcoin-miner" ]]; then
        log_error "Исполняемый файл не найден"
        exit 1
    fi
    
    # Тест справки
    if ./qubitcoin-miner --help >/dev/null 2>&1; then
        log_success "[✅] Базовый тест пройден"
    else
        log_error "[❌] Базовый тест не пройден"
        exit 1
    fi
    
    # Проверяем наличие qhash алгоритма
    if ./qubitcoin-miner --help | grep -q "qhash"; then
        log_success "[✅] Алгоритм qhash найден"
    else
        log_warning "[⚠️] Алгоритм qhash не найден в справке"
    fi
}

# Показать использование
show_usage() {
    cat << EOF
QHash Miner Build Script

Использование: $0 [ОПЦИЯ]

ОПЦИИ:
    clean      Очистить сборку
    rebuild    Полная пересборка (clean + build)
    install    Установить после сборки
    test       Тестировать собранный бинарный файл
    help       Показать эту справку

ПЕРЕМЕННЫЕ ОКРУЖЕНИЯ:
    CUDA_PATH         Путь к CUDA (автоопределение если не задан)
    CUQUANTUM_ROOT    Путь к cuQuantum (по умолчанию: /home/henuzzz/cuquantum-linux-x86_64-25.06.0.10_cuda12-archive)

ПРИМЕРЫ:
    $0                    # Обычная сборка
    $0 rebuild            # Полная пересборка
    $0 clean              # Только очистка
    CUDA_PATH=/opt/cuda $0 # Сборка с custom CUDA путем

EOF
}

# Основная функция
main() {
    log_info "=== QHash Miner Build Script ==="
    
    # Обрабатываем аргументы
    case "${1:-build}" in
        "help"|"-h"|"--help")
            show_usage
            exit 0
            ;;
        "clean")
            clean_build
            exit 0
            ;;
        "test")
            test_build
            exit 0
            ;;
        "rebuild")
            setup_paths
            check_dependencies
            clean_build
            setup_autotools
            configure_build
            build_project
            test_build
            ;;
        "install")
            if [[ ! -f "qubitcoin-miner" ]]; then
                log_error "Сначала выполните сборку"
                exit 1
            fi
            install_binary
            exit 0
            ;;
        "build"|"")
            setup_paths
            check_dependencies
            setup_autotools
            configure_build
            build_project
            test_build
            ;;
        *)
            log_error "Неизвестная опция: $1"
            show_usage
            exit 1
            ;;
    esac
    
    log_success "=== Готово! ==="
    log_info "Для запуска бенчмарка: ./qubitcoin-miner -a qhash --benchmark -t 1"
    log_info "Для запуска майнинга на LuckyPool: ./qubitcoin-miner -a qhash -o qubitcoin.luckypool.io:8611 -u ВАШ_АДРЕСС.ИМЯ_ВОРКЕРА -t КОЛ_ВО_ЯДЕР_CPU"
    log_info "Для повторной сборки: $0 install"
}

# Запускаем если скрипт выполняется напрямую
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi 