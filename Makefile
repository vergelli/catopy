#* =============================================================================
#* CATOPY - CUDA Accelerated Tensor Operations
#* =============================================================================

RED    = \033[31m
GREEN  = \033[32m
YELLOW = \033[33m
BLUE   = \033[34m
PURPLE = \033[35m
CYAN   = \033[36m
WHITE  = \033[37m
BOLD   = \033[1m
RESET  = \033[0m

#* Variables del proyecto
PROJECT_NAME = cato
BUILD_DIR = build
PROFILING_PATH = tests/profiling/evidences
VENV_ACTIVATE = . .venv/bin/activate

#* =============================================================================
#* COMANDOS PRINCIPALES
#* =============================================================================

.PHONY: help all clean build install test test-unit test-cuda test-coverage test-watch

#* Comando por defecto - muestra ayuda
help:
	@echo "$(BOLD)$(CYAN)============================================$(RESET)"
	@echo "$(BOLD)$(CYAN)  CATOPY - Makefile de Desarrollo$(RESET)"
	@echo "$(BOLD)$(CYAN)============================================$(RESET)"
	@echo ""
	@echo "$(BOLD)$(GREEN)Comandos disponibles:$(RESET)"
	@echo ""
	@echo "$(BOLD)$(YELLOW)  all$(RESET)      - Ejecuta todo el flujo: configurar + compilar + instalar"
	@echo "$(BOLD)$(YELLOW)  test-all$(RESET) - Ejecuta todo + tests unitarios"
	@echo "$(BOLD)$(YELLOW)  install-dependencies$(RESET) - Instala dependencias del sistema (spdlog, CUDA, etc.)"
	@echo "$(BOLD)$(YELLOW)  config$(RESET)   - Configura el proyecto con meson (solo primera vez o cambios de config)"
	@echo "$(BOLD)$(YELLOW)  build$(RESET)    - Compila el código con ninja (después de cambios de código)"
	@echo "$(BOLD)$(YELLOW)  install$(RESET)  - Instala el módulo compilado con uv pip"
	@echo "$(BOLD)$(YELLOW)  test$(RESET)     - Prueba el módulo instalado"
	@echo ""
	@echo "$(BOLD)$(CYAN)Testing por Tecnología:$(RESET)"
	@echo "$(BOLD)$(YELLOW)  test-frontend$(RESET) - Tests unitarios Python (pytest)"
	@echo "$(BOLD)$(YELLOW)  test-backend$(RESET) - Tests unitarios C++/CUDA"
	@echo "$(BOLD)$(YELLOW)  test-profiling$(RESET) - Tests de rendimiento Python"
	@echo "$(BOLD)$(YELLOW)  test-integration$(RESET) - Tests de integración Robot Framework"
	@echo "$(BOLD)$(YELLOW)  test-all$(RESET) - Todos los tests organizados"
	@echo ""
	@echo "$(BOLD)$(CYAN)Testing Especializado:$(RESET)"
	@echo "$(BOLD)$(YELLOW)  test-unit$(RESET) - Alias para test-frontend"
	@echo "$(BOLD)$(YELLOW)  test-cuda$(RESET) - Tests que requieren CUDA"
	@echo "$(BOLD)$(YELLOW)  test-coverage$(RESET) - Reporte de cobertura"
	@echo "$(BOLD)$(YELLOW)  test-fast$(RESET) - Tests rápidos (solo frontend)"
	@echo "$(BOLD)$(YELLOW)  test-watch$(RESET) - Tests en modo watch"
	@echo "$(BOLD)$(YELLOW)  test-file$(RESET) - Tests de archivo específico"
	@echo ""
	@echo "$(BOLD)$(CYAN)Profiling:$(RESET)"
	@echo "$(BOLD)$(YELLOW)  profile-quick$(RESET) - Profiling rápido (operaciones básicas)"
	@echo "$(BOLD)$(YELLOW)  profile-full$(RESET) - Profiling completo (todas las trazas)"
	@echo "$(BOLD)$(YELLOW)  profile-memory$(RESET) - Profiling enfocado en memoria"
	@echo "$(BOLD)$(YELLOW)  profile-memory-transfer$(RESET) - Profiling específico para transferencias HOST↔GPU"
	@echo "$(BOLD)$(YELLOW)  profile-auto-open$(RESET) - Profiling + abre Nsight Systems automáticamente"
	@echo "$(BOLD)$(YELLOW)  profile-quick-open$(RESET) - Profiling rápido + abre Nsight Systems"
	@echo "$(BOLD)$(YELLOW)  profile-full-open$(RESET) - Profiling completo + abre Nsight Systems"
	@echo "$(BOLD)$(YELLOW)  profile-open$(RESET) - Abre último reporte de profiling"
	@echo "$(BOLD)$(YELLOW)  profile-list$(RESET) - Lista reportes disponibles"
	@echo "$(BOLD)$(YELLOW)  profile-clean$(RESET) - Limpia reportes de profiling"
	@echo ""
	@echo "$(BOLD)$(CYAN)Utilidades:$(RESET)"
	@echo "$(BOLD)$(YELLOW)  clean$(RESET)    - Limpia archivos de build"
	@echo "$(BOLD)$(YELLOW)  rebuild$(RESET)  - Limpia y recompila todo"
	@echo "$(BOLD)$(YELLOW)  quick$(RESET)    - Compila + verifica (cambios frecuentes)"
	@echo "$(BOLD)$(YELLOW)  check$(RESET)    - Solo compila (verificar que compila)"
	@echo "$(BOLD)$(YELLOW)  info$(RESET)     - Muestra información del proyecto"
	@echo ""
	@echo "$(BOLD)$(BLUE)Flujo típico de desarrollo:$(RESET)"
	@echo "  1. Primera vez en el sistema → make install-dependencies"
	@echo "  2. Cambias código fuente → make build"
	@echo "  3. Cambias configuración → make config"
	@echo "  4. Quieres probar → make install"
	@echo "  5. Ejecutar tests → make test-frontend"
	@echo "  6. Todo desde cero → make all"
	@echo "  7. Todo + tests → make test-all"
	@echo "  8. Profiling → make profile-memory-transfer"
	@echo "  9. Profiling + Nsight → make profile-auto-open"
	@echo "  10. Desarrollo completo → make dev-profile"
	@echo ""

#* Comando principal - ejecuta todo el flujo
all: config build install
	@echo "$(BOLD)$(GREEN)Todo completado exitosamente!$(RESET)"
	@echo "$(CYAN)El módulo $(BOLD)$(PROJECT_NAME)$(RESET)$(CYAN) está listo para usar.$(RESET)"

#* Comando completo con tests
test-all: config build install test-frontend test-backend test-profiling
	@echo "$(BOLD)$(GREEN)Todo completado exitosamente con tests!$(RESET)"
	@echo "$(CYAN)El módulo $(BOLD)$(PROJECT_NAME)$(RESET)$(CYAN) está listo y probado.$(RESET)"

#& =============================================================================
#& INSTALACIÓN DE DEPENDENCIAS DEL SISTEMA
#& =============================================================================

#* Instala todas las dependencias del sistema necesarias para el proyecto
install-dependencies:
	@echo "$(BOLD)$(BLUE)Instalando dependencias del sistema...$(RESET)"
	@echo "$(CYAN)Este paso instala spdlog, CUDA y otras dependencias del sistema$(RESET)"
	@echo "$(YELLOW)Se requiere sudo para instalar paquetes del sistema$(RESET)"
	@sudo apt update
	@sudo apt install -y libspdlog-dev
	@echo "$(BOLD)$(GREEN)Dependencias del sistema instaladas$(RESET)"
	@echo "$(CYAN)Ahora puedes ejecutar 'make config' para configurar el proyecto$(RESET)"

#& =============================================================================
#& CONFIGURACIÓN DEL PROYECTO
#& =============================================================================

#* Configura el proyecto con meson (solo cuando cambias configuración)
config: install-dependencies
	@echo "$(BOLD)$(BLUE)Configurando proyecto con meson...$(RESET)"
	@echo "$(CYAN)Este paso detecta compiladores, dependencias y genera archivos de configuración$(RESET)"
	@$(VENV_ACTIVATE) && meson $(BUILD_DIR)
	@echo "$(BOLD)$(GREEN)Configuración completada$(RESET)"

#& =============================================================================
#& COMPILACIÓN DEL CÓDIGO
#& =============================================================================

#* Compila el código con ninja (después de cada cambio de código)
build:
	@echo "$(BOLD)$(BLUE)Compilando código con ninja...$(RESET)"
	@echo "$(CYAN)Este paso compila todo el código C++/CUDA y genera el módulo Python$(RESET)"
	@$(VENV_ACTIVATE) && ninja -C $(BUILD_DIR)
	@echo "$(BOLD)$(GREEN)Compilación completada$(RESET)"

#& =============================================================================
#& INSTALACIÓN Y PRUEBAS
#& =============================================================================

#* Instala el módulo compilado
install:
	@echo "$(BOLD)$(BLUE)Instalando módulo compilado...$(RESET)"
	@echo "$(CYAN)Este paso instala el módulo compilado en tu entorno Python$(RESET)"
	@$(VENV_ACTIVATE) && uv pip install .
	@echo "$(BOLD)$(GREEN)Instalación completada$(RESET)"
	@echo "$(CYAN)El módulo $(BOLD)$(PROJECT_NAME)$(RESET)$(CYAN) está disponible globalmente en tu entorno virtual$(RESET)"


#& =============================================================================
#& LIMPIEZA Y MANTENIMIENTO
#& =============================================================================

#* Limpia archivos de build
clean:
	@echo "$(BOLD)$(YELLOW)Limpiando archivos de build...$(RESET)"
	@rm -rf $(BUILD_DIR)
	@echo "$(BOLD)$(GREEN)Limpieza completada$(RESET)"

#* Reconstruye todo desde cero
rebuild: clean all
	@echo "$(BOLD)$(GREEN)Reconstrucción completada$(RESET)"

#& =============================================================================
#& COMANDOS DE DESARROLLO RÁPIDO
#& =============================================================================

#* Compila e instala rápidamente (para cambios de código frecuentes)
quick: build install
	@echo "$(BOLD)$(GREEN)Cambios aplicados rápidamente!$(RESET)"

#* Solo compila (para verificar que compila)
check: build
	@echo "$(BOLD)$(GREEN)Código compila correctamente$(RESET)"

#& =============================================================================
#& INFORMACIÓN DEL PROYECTO
#& =============================================================================

#* Muestra información del proyecto
info:
	@echo "$(BOLD)$(CYAN)============================================$(RESET)"
	@echo "$(BOLD)$(CYAN)  INFORMACIÓN DEL PROYECTO$(RESET)"
	@echo "$(BOLD)$(CYAN)============================================$(RESET)"
	@echo "$(BOLD)Nombre:$(RESET) $(PROJECT_NAME)"
	@echo "$(BOLD)Directorio de build:$(RESET) $(BUILD_DIR)"
	@echo "$(BOLD)Entorno virtual:$(RESET) .venv/"
	@echo "$(BOLD)Compilador CUDA:$(RESET) nvcc"
	@echo "$(BOLD)Build system:$(RESET) meson + ninja"
	@echo "$(BOLD)Package manager:$(RESET) uv"
	@echo ""

#& =============================================================================
#& NOTAS IMPORTANTES
#& =============================================================================
#* 
#* FLUJO DE DESARROLLO:
#*   1. make install-dependencies → Solo la primera vez (instala spdlog, CUDA, etc.)
#*   2. make config    → Solo la primera vez o cambios de configuración
#*   3. make build     → Después de cada cambio de código
#*   4. make install   → Para probar el módulo
#*   5. make all       → Para hacer todo desde cero
#
#* CUÁNDO USAR CADA COMANDO:
#*   - install-dependencies: Primera vez en el sistema o cambio de dependencias
#*   - config: Cambias meson.build, pyproject.toml, o agregas/quitas archivos
#*   - build:  Cambias código fuente (.cu, .cpp, .hpp, .cuh)
#*   - install: Quieres probar el módulo en Python
#
#* PROBLEMAS COMUNES:
#*   - Si build falla → Verifica que config se ejecutó correctamente
#*   - Si install falla → Verifica que build se ejecutó correctamente
#*   - Si hay errores de dependencias → Ejecuta make config
#
#* =============================================================================

#& ===== PROFILING TARGETS =====================================================
.PHONY: profile profile-quick profile-full profile-memory profile-kernels profile-memory-transfer profile-auto-open

#* Create profiling directory
profiling:
	mkdir -p $(PROFILING_PATH)
	@echo "Created profiling directory: $(PROFILING_PATH)"

#* Quick profiling (basic CUDA operations)
profile-quick: profiling
	@echo "Quick profiling (basic CUDA operations)..."
	nsys profile --stats=true --trace=cuda,nvtx,osrt \
		--output=$(PROFILING_PATH)/profile_quick_$(shell date +%Y%m%d_%H%M%S) \
		python tests/profiling/test_basic_profiling.py

#* Full profiling (all available traces)
profile-full: profiling
	@echo "Full profiling (all available traces)..."
	nsys profile --stats=true --trace=cuda,nvtx,osrt,cudnn,cublas \
		--cuda-memory-usage=true --cuda-graph-trace=node \
		--cudabacktrace=all --backtrace=dwarf \
		--sample=process-tree --cpuctxsw=process-tree \
		--output=$(PROFILING_PATH)/profile_full_$(shell date +%Y%m%d_%H%M%S) \
		python tests/profiling/test_complete_profiling.py

#* Memory-focused profiling
profile-memory: profiling
	@echo "Memory-focused profiling..."
	nsys profile --stats=true --trace=cuda,nvtx,osrt \
		--cuda-memory-usage=true --cuda-um-cpu-page-faults=true \
		--cuda-um-gpu-page-faults=true \
		--output=$(PROFILING_PATH)/profile_memory_$(shell date +%Y%m%d_%H%M%S) \
		python tests/profiling/test_memory_profiling.py

#* Memory transfer profiling (SPECIFIC FOR MEMORY TRANSFER VISUALIZATION)
profile-memory-transfer: profiling
	@echo "Memory Transfer Profiling (focused on HOST↔GPU transfers)..."
	@echo "This will show:"
	@echo "   - cudaMemcpy operations (HOST→GPU, GPU→HOST)"
	@echo "   - Memory allocation patterns"
	@echo "   - Transfer timing differences"
	@echo "   - Lazy copy behavior"
	nsys profile --stats=true --trace=cuda,nvtx,osrt \
		--cuda-memory-usage=true \
		--cuda-um-cpu-page-faults=true \
		--cuda-um-gpu-page-faults=true \
		--sample=cpu --cpuctxsw=process-tree \
		--output=$(PROFILING_PATH)/profile_memory_transfer_$(shell date +%Y%m%d_%H%M%S) \
		python tests/profiling/test_memory_transfer_visualization.py

#* Kernel profiling (when we have CUDA kernels)
profile-kernels: profiling
	@echo "Kernel profiling..."
	nsys profile --stats=true --trace=cuda,nvtx,osrt \
		--cuda-memory-usage=true \
		--output=$(PROFILING_PATH)/profile_kernels_$(shell date +%Y%m%d_%H%M%S) \
		python tests/profiling/test_kernel_profiling.py

#* AUTOMATIC WORKFLOW: Profile + Open Nsight Systems
profile-auto-open: profile-memory-transfer
	@echo "Profiling completed! Opening Nsight Systems automatically..."
	@make profile-open

#* AUTOMATIC WORKFLOW: Quick Profile + Open
profile-quick-open: profile-quick
	@echo "Quick profiling completed! Opening Nsight Systems automatically..."
	@make profile-open

#* AUTOMATIC WORKFLOW: Full Profile + Open
profile-full-open: profile-full
	@echo "Full profiling completed! Opening Nsight Systems automatically..."
	@make profile-open

#* Open latest profiling report
profile-open:
	@echo "Opening latest profiling report..."
	@latest_report=$$(ls -t $(PROFILING_PATH)/*.nsys-rep | head -1); \
	if [ -n "$$latest_report" ]; then \
		echo "Opening: $$latest_report"; \
		echo "🔍 Look for these key operations in the timeline:"; \
		echo "   - cudaMemcpy (HOST→GPU): Vector transfers to GPU"; \
		echo "   - cudaMemcpy (GPU→HOST): Vector transfers back to HOST"; \
		echo "   - Memory allocations: GPU buffer creation"; \
		echo "   - NVTX markers: Test function boundaries"; \
		nsys-ui "$$latest_report" & \
	else \
		echo "No profiling reports found. Run 'make profile-memory-transfer' first."; \
	fi

#* List all profiling reports
profile-list:
	@echo "Available profiling reports:"
	@ls -la $(PROFILING_PATH)/*.nsys-rep 2>/dev/null || echo "No reports found"

#* Clean profiling reports
profile-clean:
	@echo "$(BOLD)$(YELLOW)Limpiando reportes de profiling...$(RESET)"
	rm -rf $(PROFILING_PATH)/*.nsys-rep $(PROFILING_PATH)/*.sqlite
	@echo "$(BOLD)$(GREEN)Reportes de profiling limpiados$(RESET)"

#& =============================================================================
#& TESTING ORGANIZADO POR CAPAS
#& =============================================================================

.PHONY: test-frontend test-backend test-profiling test-integration test-all test-unit test-cuda test-coverage test-watch test-fast test-file

#* Frontend tests (Python bindings) - Incluye coverage
test-frontend:
	@echo "$(BOLD)$(BLUE)Ejecutando tests frontend (Python bindings)...$(RESET)"
	@$(VENV_ACTIVATE) && python -m pytest tests/frontend/ -v --tb=short
	@echo "$(BOLD)$(GREEN)Tests frontend completados$(RESET)"

#* Backend tests (C++/CUDA)
test-backend:
	@echo "$(BOLD)$(BLUE)Ejecutando tests backend (C++/CUDA)...$(RESET)"
	@$(VENV_ACTIVATE) && python -m pytest tests/backend/ -v --tb=short
	@echo "$(BOLD)$(GREEN)Tests backend completados$(RESET)"

#* Profiling tests (rendimiento Python)
test-profiling:
	@echo "$(BOLD)$(BLUE)Ejecutando tests de profiling...$(RESET)"
	@$(VENV_ACTIVATE) && python -m pytest tests/profiling/ -v --tb=short
	@echo "$(BOLD)$(GREEN)Tests de profiling completados$(RESET)"

#* Integration tests (Robot Framework)
test-integration:
	@echo "$(BOLD)$(BLUE)Ejecutando tests de integración...$(RESET)"
	@echo "$(CYAN)Tests de integración con Robot Framework (futuro)$(RESET)"
	@echo "$(YELLOW)Comando no implementado aún$(RESET)"

#* Tests unitarios (alias para frontend)
test-unit: test-frontend
	@echo "$(CYAN)test-unit es un alias para test-frontend$(RESET)"

#* Tests que requieren CUDA
test-cuda:
	@echo "$(BOLD)$(BLUE)Ejecutando tests que requieren CUDA...$(RESET)"
	@$(VENV_ACTIVATE) && python -m pytest tests/ -v -m cuda --tb=short
	@echo "$(BOLD)$(GREEN)Tests CUDA completados$(RESET)"

#* Tests con reporte de cobertura
test-coverage:
	@echo "$(BOLD)$(BLUE)Ejecutando tests con cobertura...$(RESET)"
	@$(VENV_ACTIVATE) && python -m pytest tests/frontend/ -v --cov=$(PROJECT_NAME) --cov-report=html --cov-report=term-missing
	@echo "$(BOLD)$(GREEN)Reporte de cobertura generado$(RESET)"
	@echo "$(CYAN)📁 Abre htmlcov/index.html para ver el reporte completo$(RESET)"

#* Tests en modo watch
test-watch:
	@echo "$(BOLD)$(BLUE)Ejecutando tests en modo watch...$(RESET)"
	@echo "$(YELLOW)Presiona Ctrl+C para detener$(RESET)"
	@$(VENV_ACTIVATE) && python -m pytest tests/frontend/ -v --tb=short --watch

#* Tests rápidos (solo frontend, sin CUDA)
test-fast:
	@echo "$(BOLD)$(BLUE)⚡ Ejecutando tests rápidos...$(RESET)"
	@$(VENV_ACTIVATE) && python -m pytest tests/frontend/ -v -m "not cuda" --tb=short
	@echo "$(BOLD)$(GREEN)Tests rápidos completados$(RESET)"

#* Tests específicos por archivo
test-file:
	@echo "$(BOLD)$(BLUE)Ejecutando tests específicos...$(RESET)"
	@if [ -z "$(FILE)" ]; then \
		echo "$(RED)Error: Especifica el archivo con FILE=path/to/test.py$(RESET)"; \
		echo "$(YELLOW)Ejemplo: make test-file FILE=tests/frontend/test_basic_functionality.py$(RESET)"; \
		exit 1; \
	fi
	@$(VENV_ACTIVATE) && python -m pytest $(FILE) -v --tb=short
	@echo "$(BOLD)$(GREEN)Tests del archivo completados$(RESET)"

#* Todos los tests organizados (alias para test-all principal)
test-all-organized: test-frontend test-backend test-profiling
	@echo "$(BOLD)$(GREEN)Todos los tests organizados completados!$(RESET)"

#& ===== DEVELOPMENT WORKFLOW =====
.PHONY: dev-test dev-profile

# Development testing workflow
dev-test: build test-all
	@echo "Development testing completed!"

# Development profiling workflow
dev-profile: build profile-full profile-open
	@echo "Development profiling completed!"
