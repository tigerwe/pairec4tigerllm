package com.pairec.flink;

import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.functions.AggregateFunction;
import org.apache.flink.api.common.serialization.SimpleStringSchema;
import org.apache.flink.connector.kafka.sink.KafkaRecordSerializationSchema;
import org.apache.flink.connector.kafka.sink.KafkaSink;
import org.apache.flink.connector.kafka.source.KafkaSource;
import org.apache.flink.connector.kafka.source.enumerator.initializer.OffsetsInitializer;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.windowing.assigners.SlidingEventTimeWindows;
import org.apache.flink.streaming.api.windowing.time.Time;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.fasterxml.jackson.databind.node.ArrayNode;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;

/**
 * 用户特征实时计算作业.
 * 
 * 功能：
 * 1. 消费用户点击流 (Kafka topic: user-clicks)
 * 2. 滑动窗口聚合最近 N 个点击物品
 * 3. 输出到 Kafka topic: user-features
 * 
 * 使用方式:
 *   flink run -c com.pairec.flink.UserFeatureJob target/user-feature-job-1.0-SNAPSHOT.jar \
 *     --kafka.bootstrap.servers localhost:9092
 */
public class UserFeatureJob {
    
    private static final ObjectMapper mapper = new ObjectMapper();
    
    public static void main(String[] args) throws Exception {
        // 解析参数
        String kafkaServers = "localhost:9092";
        String inputTopic = "user-clicks";
        String outputTopic = "user-features";
        int windowSizeMinutes = 60;
        int slideMinutes = 1;
        int maxHistory = 50;  // 保留最近50个点击
        
        // 从 args 解析参数
        for (String arg : args) {
            if (arg.startsWith("--kafka.bootstrap.servers=")) {
                kafkaServers = arg.substring("--kafka.bootstrap.servers=".length());
            } else if (arg.startsWith("--input-topic=")) {
                inputTopic = arg.substring("--input-topic=".length());
            } else if (arg.startsWith("--output-topic=")) {
                outputTopic = arg.substring("--output-topic=".length());
            } else if (arg.startsWith("--max-history=")) {
                maxHistory = Integer.parseInt(arg.substring("--max-history=".length()));
            }
        }
        
        System.out.println("Starting UserFeatureJob...");
        System.out.println("  Kafka: " + kafkaServers);
        System.out.println("  Input: " + inputTopic);
        System.out.println("  Output: " + outputTopic);
        System.out.println("  Max History: " + maxHistory);
        
        final StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(1);
        
        // 1. 创建 Kafka Source
        KafkaSource<String> source = KafkaSource.<String>builder()
            .setBootstrapServers(kafkaServers)
            .setTopics(inputTopic)
            .setGroupId("flink-user-feature-processor")
            .setStartingOffsets(OffsetsInitializer.latest())
            .setValueOnlyDeserializer(new SimpleStringSchema())
            .build();
        
        // 2. 创建 Kafka Sink
        KafkaSink<String> sink = KafkaSink.<String>builder()
            .setBootstrapServers(kafkaServers)
            .setRecordSerializer(KafkaRecordSerializationSchema.builder()
                .setTopic(outputTopic)
                .setValueSerializationSchema(new SimpleStringSchema())
                .build())
            .build();
        
        // 3. 构建数据流
        DataStream<String> inputStream = env
            .fromSource(source, WatermarkStrategy.forBoundedOutOfOrderness(Duration.ofSeconds(5)), "Kafka Source")
            .map(json -> {
                // 解析点击事件
                ClickEvent event = mapper.readValue(json, ClickEvent.class);
                return event;
            })
            .returns(ClickEvent.class)
            .keyBy(ClickEvent::getUserId)
            .window(SlidingEventTimeWindows.of(
                Time.minutes(windowSizeMinutes), 
                Time.minutes(slideMinutes)
            ))
            .aggregate(new ClickHistoryAggregator(maxHistory))
            .map(features -> {
                // 序列化为 JSON
                return mapper.writeValueAsString(features);
            });
        
        // 4. 输出到 Kafka
        inputStream.sinkTo(sink);
        
        // 5. 执行
        env.execute("UserFeatureJob");
    }
    
    /**
     * 点击事件 POJO
     */
    public static class ClickEvent {
        private String userId;
        private Long itemId;
        private Long timestamp;
        private String category;
        
        public ClickEvent() {}
        
        public String getUserId() { return userId; }
        public void setUserId(String userId) { this.userId = userId; }
        
        public Long getItemId() { return itemId; }
        public void setItemId(Long itemId) { this.itemId = itemId; }
        
        public Long getTimestamp() { return timestamp; }
        public void setTimestamp(Long timestamp) { this.timestamp = timestamp; }
        
        public String getCategory() { return category; }
        public void setCategory(String category) { this.category = category; }
    }
    
    /**
     * 用户特征 POJO
     */
    public static class UserFeatures {
        private String userId;
        private List<Long> clickHistory;
        private String updateTime;
        private double realtimeCtr;
        private List<String> shortTermTags;
        
        public UserFeatures() {
            this.clickHistory = new ArrayList<>();
            this.shortTermTags = new ArrayList<>();
        }
        
        public String getUserId() { return userId; }
        public void setUserId(String userId) { this.userId = userId; }
        
        public List<Long> getClickHistory() { return clickHistory; }
        public void setClickHistory(List<Long> clickHistory) { this.clickHistory = clickHistory; }
        
        public String getUpdateTime() { return updateTime; }
        public void setUpdateTime(String updateTime) { this.updateTime = updateTime; }
        
        public double getRealtimeCtr() { return realtimeCtr; }
        public void setRealtimeCtr(double realtimeCtr) { this.realtimeCtr = realtimeCtr; }
        
        public List<String> getShortTermTags() { return shortTermTags; }
        public void setShortTermTags(List<String> shortTermTags) { this.shortTermTags = shortTermTags; }
    }
    
    /**
     * 点击历史聚合器
     * 保留最近 N 个点击物品
     */
    public static class ClickHistoryAggregator 
        implements AggregateFunction<ClickEvent, List<Long>, UserFeatures> {
        
        private final int maxHistory;
        
        public ClickHistoryAggregator(int maxHistory) {
            this.maxHistory = maxHistory;
        }
        
        @Override
        public List<Long> createAccumulator() {
            return new ArrayList<>();
        }
        
        @Override
        public List<Long> add(ClickEvent value, List<Long> accumulator) {
            accumulator.add(value.getItemId());
            // 保留最近 N 个
            if (accumulator.size() > maxHistory) {
                return new ArrayList<>(accumulator.subList(
                    accumulator.size() - maxHistory, 
                    accumulator.size()
                ));
            }
            return accumulator;
        }
        
        @Override
        public UserFeatures getResult(List<Long> accumulator) {
            UserFeatures features = new UserFeatures();
            // 这里无法直接获取 userId，需要在 keyBy 后处理
            // 简化起见，userId 由下游填充
            features.setClickHistory(new ArrayList<>(accumulator));
            features.setUpdateTime(java.time.Instant.now().toString());
            features.setRealtimeCtr(0.0);  // 简化处理
            return features;
        }
        
        @Override
        public List<Long> merge(List<Long> a, List<Long> b) {
            List<Long> merged = new ArrayList<>(a);
            merged.addAll(b);
            if (merged.size() > maxHistory) {
                return new ArrayList<>(merged.subList(
                    merged.size() - maxHistory, 
                    merged.size()
                ));
            }
            return merged;
        }
    }
}
