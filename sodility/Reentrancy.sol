// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title Reentrancy Vulnerability Example
 * @notice This contract is intentionally vulnerable to demonstrate a reentrancy attack.
 */
contract VulnerableBank {
    mapping(address => uint256) public balances;

    // Deposit ether into the contract
    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }

    // Vulnerable withdraw function (Reentrancy!)
    function withdraw(uint256 _amount) public {
        require(balances[msg.sender] >= _amount, "Insufficient balance");

        // ❌ External call before state update
        (bool sent, ) = msg.sender.call{value: _amount}("");
        require(sent, "Failed to send Ether");

        // State update occurs after external call (dangerous)
        balances[msg.sender] -= _amount;
    }

    // View balance
    function getBalance() public view returns (uint256) {
        return balances[msg.sender];
    }
}
