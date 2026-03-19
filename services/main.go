// main.go
//
// pairec4tigerllm 服务入口.
// 基于 pairec 框架的推荐服务，集成生成式召回.

package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/alibaba/pairec/v2"
	"github.com/alibaba/pairec/v2/log"

	_ "pairec4tigerllm/services/recall"
)

var (
	configPath = flag.String("config", "./configs/pairec_config.json", "Path to config file")
	port       = flag.Int("port", 8080, "HTTP service port")
)

func main() {
	flag.Parse()

	// 检查配置文件
	if _, err := os.Stat(*configPath); os.IsNotExist(err) {
		log.Error(fmt.Sprintf("Config file not found: %s", *configPath))
		os.Exit(1)
	}

	log.Info(fmt.Sprintf("Starting pairec4tigerllm service..."))
	log.Info(fmt.Sprintf("Config: %s, Port: %d", *configPath, *port))

	// 创建 pairec 应用
	app := pairec.NewApp()

	// 加载配置
	if err := app.LoadConfig(*configPath); err != nil {
		log.Error(fmt.Sprintf("Failed to load config: %v", err))
		os.Exit(1)
	}

	// 设置端口
	app.SetPort(*port)

	// 启动服务
	if err := app.Start(); err != nil {
		log.Error(fmt.Sprintf("Failed to start service: %v", err))
		os.Exit(1)
	}
}
