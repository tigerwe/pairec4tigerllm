// main.go
//
// pairec4tigerllm 服务入口.
// 基于 pairec 框架的推荐服务，集成生成式召回.

package main

import (
	"fmt"
	"os"

	"github.com/alibaba/pairec/v2"
	"github.com/alibaba/pairec/v2/recconf"
	"github.com/alibaba/pairec/v2/service/recall"

	myrecall "pairec4tigerllm/services/recall"
)

func main() {
	fmt.Println("========PAIREC STARING===============")
	
	// 设置配置路径（环境变量或默认）
	configPath := os.Getenv("CONFIG_PATH")
	if configPath == "" {
		configPath = "../configs/pairec_config.json"
	}

	fmt.Printf("Loading config from: %s\n", configPath)

	// 1. 先加载配置
	if err := recconf.LoadConfig(configPath); err != nil {
		fmt.Printf("Failed to load config: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Config loaded. Found %d recall configs\n", len(recconf.Config.RecallConfs))

	// 2. 注册自定义召回（必须在 pairec.Run 之前）
	for _, conf := range recconf.Config.RecallConfs {
		fmt.Printf("Checking recall: name=%s, type=%s\n", conf.Name, conf.RecallType)
		if conf.RecallType == "GenerativeRecall" {
			fmt.Printf("Registering GenerativeRecall: %s\n", conf.Name)
			instance := myrecall.NewGenerativeRecall(conf)
			recall.RegisterRecall(conf.Name, instance)
			fmt.Printf("[DEBUG] Registered recall: name=%s, instance=%p\n", conf.Name, instance)
		}
	}
	
	// 验证注册是否成功
	fmt.Println("[DEBUG] All recalls registered, starting pairec...")

	// 3. 启动 pairec
	fmt.Println("Starting pairec service...")
	pairec.Run()
}
