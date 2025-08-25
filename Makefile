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
	@echo "$(BOLD)$(YELLOW)  config$(RESET)   - Configura el proyecto con meson (solo primera vez o cambios de config)"
	@echo "$(BOLD)$(YELLOW)  build$(RESET)    - Compila el código con ninja (después de cambios de código)"
	@echo "$(BOLD)$(YELLOW)  install$(RESET)  - Instala el módulo compilado con uv pip"
	@echo "$(BOLD)$(YELLOW)  test$(RESET)     - Prueba el módulo instalado"
	@echo ""
	@echo "$(BOLD)$(CYAN)Testing:$(RESET)"
	@echo "$(BOLD)$(YELLOW)  test-unit$(RESET) - Ejecuta tests unitarios con pytest"
	@echo "$(BOLD)$(YELLOW)  test-cuda$(RESET) - Ejecuta tests que requieren CUDA"
	@echo "$(BOLD)$(YELLOW)  test-coverage$(RESET) - Ejecuta tests con reporte de cobertura"
	@echo "$(BOLD)$(YELLOW)  test-fast$(RESET) - Ejecuta tests rápidos (sin CUDA)"
	@echo "$(BOLD)$(YELLOW)  test-watch$(RESET) - Ejecuta tests en modo watch (requiere pytest-watch)"
	@echo "$(BOLD)$(YELLOW)  test-file$(RESET) - Ejecuta tests de un archivo específico"
	@echo ""
	@echo "$(BOLD)$(CYAN)Utilidades:$(RESET)"
	@echo "$(BOLD)$(YELLOW)  clean$(RESET)    - Limpia archivos de build"
	@echo "$(BOLD)$(YELLOW)  rebuild$(RESET)  - Limpia y recompila todo"
	@echo "$(BOLD)$(YELLOW)  quick$(RESET)    - Compila + verifica (cambios frecuentes)"
	@echo "$(BOLD)$(YELLOW)  check$(RESET)    - Solo compila (verificar que compila)"
	@echo "$(BOLD)$(YELLOW)  info$(RESET)     - Muestra información del proyecto"
	@echo ""
	@echo "$(BOLD)$(BLUE)Flujo típico de desarrollo:$(RESET)"
	@echo "  1. Cambias código fuente → make build"
	@echo "  2. Cambias configuración → make config"
	@echo "  3. Quieres probar → make install"
	@echo "  4. Ejecutar tests → make test-unit"
	@echo "  5. Todo desde cero → make all"
	@echo "  6. Todo + tests → make test-all"
	@echo ""

#* Comando principal - ejecuta todo el flujo
all: config build install
	@echo "$(BOLD)$(GREEN)✅ Todo completado exitosamente!$(RESET)"
	@echo "$(CYAN)El módulo $(BOLD)$(PROJECT_NAME)$(RESET)$(CYAN) está listo para usar.$(RESET)"

#* Comando completo con tests
test-all: config build install test-unit
	@echo "$(BOLD)$(GREEN)✅ Todo completado exitosamente con tests!$(RESET)"
	@echo "$(CYAN)El módulo $(BOLD)$(PROJECT_NAME)$(RESET)$(CYAN) está listo y probado.$(RESET)"

#& =============================================================================
#& CONFIGURACIÓN DEL PROYECTO
#& =============================================================================

#* Configura el proyecto con meson (solo cuando cambias configuración)
config:
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
	@echo "$(BOLD)$(BLUE)📦 Instalando módulo compilado...$(RESET)"
	@echo "$(CYAN)Este paso instala el módulo compilado en tu entorno Python$(RESET)"
	@$(VENV_ACTIVATE) && uv pip install -e .
	@echo "$(BOLD)$(GREEN)✅ Instalación completada$(RESET)"
	@echo "$(CYAN)El módulo $(BOLD)$(PROJECT_NAME)$(RESET)$(CYAN) está disponible globalmente en tu entorno virtual$(RESET)"

#& =============================================================================
#& TESTING UNITARIO
#& =============================================================================

#* Ejecuta tests unitarios básicos
test-unit:
	@echo "$(BOLD)$(BLUE)🧪 Ejecutando tests unitarios...$(RESET)"
	@echo "$(CYAN)Ejecutando pytest con configuración estándar$(RESET)"
	@$(VENV_ACTIVATE) && python -m pytest tests/ -v --tb=short
	@echo "$(BOLD)$(GREEN)✅ Tests unitarios completados$(RESET)"

#* Ejecuta tests que requieren CUDA
test-cuda:
	@echo "$(BOLD)$(BLUE)🚀 Ejecutando tests CUDA...$(RESET)"
	@echo "$(CYAN)Ejecutando tests marcados con 'cuda'$(RESET)"
	@$(VENV_ACTIVATE) && python -m pytest tests/ -v -m cuda --tb=short
	@echo "$(BOLD)$(GREEN)✅ Tests CUDA completados$(RESET)"

#* Ejecuta tests con reporte de cobertura
test-coverage:
	@echo "$(BOLD)$(BLUE)📊 Ejecutando tests con cobertura...$(RESET)"
	@echo "$(CYAN)Generando reporte de cobertura de código$(RESET)"
	@$(VENV_ACTIVATE) && python -m pytest tests/ -v --cov=$(PROJECT_NAME) --cov-report=html --cov-report=term-missing
	@echo "$(BOLD)$(GREEN)✅ Reporte de cobertura generado$(RESET)"
	@echo "$(CYAN)📁 Abre htmlcov/index.html para ver el reporte completo$(RESET)"

#* Ejecuta tests en modo watch (requiere pytest-watch)
test-watch:
	@echo "$(BOLD)$(BLUE)👀 Ejecutando tests en modo watch...$(RESET)"
	@echo "$(CYAN)Los tests se ejecutarán automáticamente cuando cambies archivos$(RESET)"
	@echo "$(YELLOW)⚠️  Presiona Ctrl+C para detener$(RESET)"
	@$(VENV_ACTIVATE) && python -m pytest tests/ -v --tb=short --watch

#* Ejecuta tests rápidos (solo unitarios, sin CUDA)
test-fast:
	@echo "$(BOLD)$(BLUE)⚡ Ejecutando tests rápidos...$(RESET)"
	@echo "$(CYAN)Ejecutando tests unitarios sin tests CUDA$(RESET)"
	@$(VENV_ACTIVATE) && python -m pytest tests/ -v -m "not cuda" --tb=short
	@echo "$(BOLD)$(GREEN)✅ Tests rápidos completados$(RESET)"

#* Ejecuta tests específicos por archivo
test-file:
	@echo "$(BOLD)$(BLUE)📁 Ejecutando tests específicos...$(RESET)"
	@echo "$(CYAN)Uso: make test-file FILE=tests/devices/test_device_detection.py$(RESET)"
	@if [ -z "$(FILE)" ]; then \
		echo "$(RED)❌ Error: Especifica el archivo con FILE=path/to/test.py$(RESET)"; \
		echo "$(YELLOW)Ejemplo: make test-file FILE=tests/devices/test_device_detection.py$(RESET)"; \
		exit 1; \
	fi
	@$(VENV_ACTIVATE) && python -m pytest $(FILE) -v --tb=short
	@echo "$(BOLD)$(GREEN)✅ Tests del archivo completados$(RESET)"

#* Prueba el módulo instalado (verificación básica)
test:
	@echo "$(BOLD)$(BLUE)🔍 Probando módulo instalado...$(RESET)"
	@echo "$(CYAN)Verificando que el módulo se puede importar y usar$(RESET)"
	@$(VENV_ACTIVATE) && python -c "import $(PROJECT_NAME); print('✅ Módulo importado exitosamente')"
	@echo "$(BOLD)$(GREEN)✅ Prueba exitosa$(RESET)"

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
#* 📝 FLUJO DE DESARROLLO:
#*   1. make config    → Solo la primera vez o cambios de configuración
#*   2. make build     → Después de cada cambio de código
#*   3. make install   → Para probar el módulo
#*   4. make all       → Para hacer todo desde cero
#
#* 📝 CUÁNDO USAR CADA COMANDO:
#*   - config: Cambias meson.build, pyproject.toml, o agregas/quitas archivos
#*   - build:  Cambias código fuente (.cu, .cpp, .hpp, .cuh)
#*   - install: Quieres probar el módulo en Python
#
#* 📝 PROBLEMAS COMUNES:
#*   - Si build falla → Verifica que config se ejecutó correctamente
#*   - Si install falla → Verifica que build se ejecutó correctamente
#*   - Si hay errores de dependencias → Ejecuta make config
#
#* =============================================================================
