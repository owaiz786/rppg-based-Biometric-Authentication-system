package com.yourapp.biometric.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import java.util.List;

@Data
public class PythonResponse {
    private boolean success;
    
    @JsonProperty("is_real")
    private boolean isReal;
    
    @JsonProperty("spoof_reason")
    private String spoofReason;
    
    @JsonProperty("coherence_score")
    private double coherenceScore;
    
    private List<Double> embedding;
}